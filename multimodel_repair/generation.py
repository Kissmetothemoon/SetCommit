"""Explicit native, fixed-NFE and Fast-dLLM cache protocols.

Frozen upstream samplers are not edited. Cached protocols are named separately
from native decoding; LLaDA2 fixed-NFE intentionally disables native T2T/S-D edits.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from vendor.LLaDA.generate import add_gumbel_noise, get_num_transfer_tokens

from .data import ROOT
from .selection import POLICY, select, audit_trace

MODELS = {
    "llada_8b_instruct": ("LLaDA-8B-Instruct", "llada1", "08b83a6feb34df1a6011b80c3c00c7563e963b07"),
    "dream_v0_instruct_7b": ("Dream-v0-Instruct-7B", "dream", "05334cb9faaf763692dcf9d8737c642be2b2a6ae"),
    "llada22_mini": ("LLaDA2.2-mini", "llada2", "038922eee61174fd2e2dc6848fb6b87eb740afb5"),
    "llada21_mini": ("LLaDA2.1-mini", "llada2", "20e64e2ad21644d0e5248586ed9c942cdd45de0f"),
}
METHODS = ("topk_uniform", "dacs_v5_exchange_020", "fast_dual_fixed", "fast_dual_fixed_dacs",
           "fast_dual_threshold", "fast_dual_threshold_dacs", "native_reference")


@dataclass
class Generated:
    ids: torch.Tensor
    trace: list[dict]
    audit: dict


def schedule(length, steps):
    if not 0 < steps <= length:
        raise ValueError("fixed-NFE steps must be in 1..generation length")
    return [length // steps + int(i < length % steps) for i in range(steps)]


class Generator:
    def __init__(self, model_key: str, *, fast: bool = False):
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        from huggingface_hub.utils import disable_progress_bars
        hf_logging.disable_progress_bar()
        disable_progress_bars()
        name, self.family, self.revision = MODELS[model_key]
        if fast and self.family == "llada2":
            raise ValueError("no official Fast-dLLM LLaDA2 adapter")
        self.fast, self.model_key = fast, model_key
        path = str(ROOT / "models" / name)
        if fast:
            sys.path.insert(0, str(ROOT / "vendor/Fast-dLLM/v1" / ("llada" if self.family == "llada1" else "dream")))
            if self.family == "llada1":
                from model.modeling_llada import LLaDAModelLM
                cls = LLaDAModelLM
                self.upstream_fast = importlib.import_module("generate")
            else:
                from model.modeling_dream import DreamModel
                cls = DreamModel
                self.upstream_fast = importlib.import_module("model.generation_utils_block")
        else:
            cls = AutoModelForCausalLM if self.family == "llada2" else AutoModel
        self.model = cls.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                         torch_dtype=torch.bfloat16).to("cuda:0").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.mask = self.tokenizer.mask_token_id
        # Remote-code models do not all hydrate generation_config.json into the
        # model object. Read the pinned local file first so multi-EOS settings
        # (notably LLaDA2.2's role-end token) cannot be silently dropped.
        generation_config_path = Path(path) / "generation_config.json"
        pinned_generation = (json.loads(generation_config_path.read_text())
                             if generation_config_path.exists() else {})
        configured_eos = pinned_generation.get(
            "eos_token_id", getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        )
        self.eos = configured_eos if configured_eos is not None else self.tokenizer.eos_token_id
        if self.family == "llada1" and self.mask is None:
            # The pinned LLaDA tokenizer does not set mask_token, whereas the
            # pinned official sampler explicitly declares ID 126336.
            self.mask = 126336
        if self.mask is None:
            raise ValueError("model tokenizer does not declare mask token")
        self.nfe = 0
        original = self.model.forward
        def counted(*args, **kwargs):
            self.nfe += 1
            return original(*args, **kwargs)
        self.model.forward = counted
        self.original_forward = original

    def encode(self, prompt):
        # Dream Base and Dream Instruct are different prompt protocols.  The
        # Fast-dLLM paper script uses raw prompt+BOS for its Base checkpoint,
        # while the pinned Instruct tokenizer ships a chat template and the
        # model card's generation example uses it.  Applying the Base
        # protocol to the Instruct checkpoint causes malformed math/code
        # answers (the old path produced a zero MATH-500 gate), so keep the
        # distinction explicit.
        if self.family == "dream":
            if self.model_key == "dream_v0_instruct_7b":
                x = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=True, return_tensors="pt"
                )
            else:
                text = (self.tokenizer.bos_token or "") + prompt
                x = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        else:
            x = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                        add_generation_prompt=True, tokenize=True, return_tensors="pt")
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict):
            x = x["input_ids"]
        return x.to(self.model.device)

    def identity(self):
        return {"model": self.model_key, "revision": self.revision,
                "tokenizer_class": type(self.tokenizer).__name__,
                "chat_template": self.tokenizer.chat_template,
                "mask_token_id": self.mask, "eos_token_id": self.eos,
                "tokenizer_eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id, "dtype": "bfloat16",
                "fast_model_class": self.fast, "policy": POLICY.to_dict(),
                "prompt_protocol": ("dream_instruct_chat_template" if self.family == "dream" and self.model_key == "dream_v0_instruct_7b"
                                     else "dream_raw_prompt_with_bos" if self.family == "dream"
                                     else "chat_template_add_generation_prompt"),
                "generation_protocol": self.generation_protocol()}

    def generation_protocol(self):
        if self.family == "dream":
            return {
                # The Dream-Instruct evaluation toolkit (pinned upstream
                # protocol) uses the confidence-entropy remasking rule with
                # temperature=0.1, top_p=0.9 and the tokenizer chat template.
                # ``origin`` in generation_config.json is the generic model
                # default and is not the published Instruct eval setting.
                "algorithm": "entropy" if not self.fast else "confidence_threshold",
                "temperature": 0.1 if not self.fast else 0.0,
                "top_p": 0.9 if not self.fast else None,
                "threshold": 0.9 if self.fast else None,
                "native_reference": "Dream eval_instruct: chat_template + entropy + temperature=0.1 + top_p=0.9",
            }
        if self.family == "llada2" and self.model_key == "llada21_mini":
            return {
                "mode": "quality",
                "temperature": 0.0,
                "threshold": 0.7,
                "editing_threshold": 0.5,
                "max_post_steps": 16,
                "eos_early_stop": True,
            }
        if self.family == "llada2":
            return {
                "mode": "speed",
                "temperature": 0.0,
                "threshold": 0.5,
                "editing_threshold": 0.0,
                "max_post_steps": 16,
                "eos_early_stop": True,
            }
        return {"temperature": 0.0}

    @torch.no_grad()
    def __call__(self, prompt_ids, *, method, steps, length):
        if method not in METHODS or method.startswith("fast_") != self.fast:
            raise ValueError("method/model loader mismatch")
        dacs = method in {"dacs_v5_exchange_020", "fast_dual_fixed_dacs", "fast_dual_threshold_dacs"}
        adaptive = method in {"fast_dual_threshold", "fast_dual_threshold_dacs"} or (method == "native_reference" and self.family == "llada2")
        self.nfe = 0
        start = prompt_ids.shape[1]
        if self.family == "llada1" and not self.fast:
            if method == "native_reference":
                from official_protocol_repair_campaign.official_sampler import generate
                seq = generate(self.model, prompt_ids, steps=steps, gen_length=length,
                               block_length=32, mask_id=self.mask, eos_guard=False)
                trace = [{"net_commits": k, "new_commits": k, "exchange_count": 0}
                         for k in schedule(length, steps)]
            else:
                seq, trace = self.llada_blockwise(prompt_ids, steps, length, dacs)
            ids = seq[0, start:]
        elif self.family == "dream" and not self.fast:
            ids, trace = self.dream_native(prompt_ids, steps, length, dacs)
        elif self.family == "llada2" and method == "native_reference":
            # LLaDA2.1's published Quality Mode is materially different from
            # its Speed Mode defaults.  The previous runner hard-coded the
            # 2.2/Speed parameters for both models, which depressed the 2.1
            # GSM8K/MBPP reference and invalidated the denominator.
            if self.model_key == "llada21_mini":
                threshold, editing_threshold = 0.7, 0.5
            else:
                threshold, editing_threshold = 0.5, 0.0
            kwargs = dict(inputs=prompt_ids, gen_length=length, block_length=32,
                          threshold=threshold, editing_threshold=editing_threshold,
                          max_post_steps=16, temperature=0.0, eos_early_stop=True)
            # 2.1's steps argument is unused; never label this as fixed-NFE.
            if self.model_key == "llada22_mini":
                kwargs["steps"] = 32
            ids = self.model.generate(**kwargs)[0]
            trace = []
        elif self.family == "llada2":
            ids, trace = self.llada2_fixed(prompt_ids, steps, length, dacs)
        elif self.family == "llada1":
            ids, trace = self.llada_fast(prompt_ids, steps, length, dacs, adaptive)
        else:
            ids, trace = self.dream_fast(prompt_ids, steps, length, dacs, adaptive)
        unresolved = int((ids == self.mask).sum().item())
        audit = audit_trace(trace, expected_nfe=None if adaptive else steps, actual_nfe=self.nfe)
        audit.update(unresolved_masks=unresolved, schedule_semantics=("native_adaptive" if method == "native_reference"
                     and self.family == "llada2" else "official_block32_reference" if method == "native_reference"
                     else "confidence_threshold_adaptive" if adaptive else "controlled_fixed_total_forward"))
        if unresolved:
            audit["passed"] = False
            audit["errors"].append("unresolved output masks")
        if not audit["passed"]:
            raise RuntimeError(str(audit))
        return Generated(ids, trace, audit)

    def llada_blockwise(self, prompt, steps, length, dacs):
        """Paper-protocol LLaDA fixed-NFE control with 32-token blocks."""
        start, block_length = prompt.shape[1], 32
        if length % block_length:
            raise ValueError("LLaDA paper-protocol fixed control requires 32-token blocks")
        blocks = length // block_length
        if steps % blocks:
            raise ValueError("fixed total steps must divide the number of LLaDA blocks")
        local_steps = steps // blocks
        if not 0 < local_steps <= block_length:
            raise ValueError("invalid LLaDA per-block step budget")
        x = torch.full((1, start + length), self.mask, dtype=torch.long, device=prompt.device)
        x[:, :start] = prompt
        trace = []
        explicit_nfe = 0
        for block in range(blocks):
            s, e = start + block * block_length, start + (block + 1) * block_length
            transfers = get_num_transfer_tokens(x[:, s:e] == self.mask, local_steps)
            for local_step in range(local_steps):
                logits = self.model(x).logits
                explicit_nfe += 1
                mask_index = x == self.mask
                mask_index[:, e:] = False
                predicted = add_gumbel_noise(logits, 0.0).argmax(-1)
                probabilities = logits.to(torch.float64).softmax(-1)
                confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                # Keep raw argmax proposals at visible positions. SETCOMMIT
                # needs them to screen remask/exchange eligibility; replacing
                # them with the old token makes ``old != new`` impossible and
                # silently disables Native remasking. Only the confidence
                # ranking below is restricted to currently masked positions.
                neg_inf = torch.tensor(torch.finfo(confidence.dtype).min,
                                       device=confidence.device, dtype=confidence.dtype)
                confidence = torch.where(mask_index, confidence, neg_inf)
                # Match pinned Fast-dLLM fixed path tie behavior: sort the full
                # candidate confidence vector and take the first scheduled quota.
                _, order = torch.sort(confidence, dim=1, descending=True)
                quota = int(transfers[0, local_step].item())
                selected_global = order[0, :quota]
                baseline = (selected_global - s).to(torch.long)
                if not bool(((baseline >= 0) & (baseline < block_length)).all()):
                    raise RuntimeError("LLaDA blockwise baseline selected outside current block")
                result, row = select(logits[0, s:e], x[0, s:e], predicted[0, s:e], baseline,
                           mask_id=self.mask, step=len(trace), steps=steps, dacs=dacs,
                           global_offset=s)
                x[0, s:e] = result
                row.update(block=block, local_step=local_step, block_length=block_length,
                           cache="none", upstream_cardinality=quota)
                trace.append(row)
        self.nfe = explicit_nfe
        return x, trace

    def dream_native(self, prompt, steps, length, dacs):
        trace, state = [], {}
        native_module = importlib.import_module(type(self.model).__module__.rsplit(".", 1)[0] + ".generation_utils")
        start = prompt.shape[1]
        def logits_hook(i, x, logits):
            state["before"] = x.clone()
            return logits
        def tokens_hook(i, x, logits):
            if i is None:
                return x
            before = state["before"]
            baseline = ((before[0, start:] == self.mask) & (x[0, start:] != self.mask)).nonzero().flatten()
            # The upstream Instruct sampler may draw x0 with temperature/top_p
            # (the official setting is temperature=0.1, top_p=0.9).  Calling
            # sample_tokens again here would consume a second RNG draw and
            # silently change even the no-op baseline.  Preserve the upstream
            # sequence exactly for dacs=False; DACS mode may use a fresh
            # deterministic posterior proposal for its selector.
            if not dacs:
                committed = baseline.tolist()
                trace.append({
                    "step": i, "offset": start, "mask_count": int((before[0, start:] == self.mask).sum()),
                    "mask_after": int((x[0, start:] == self.mask).sum()),
                    "net_commits": len(committed), "new_commits": len(committed),
                    "exchange_count": 0, "exchange_cap": 0,
                    "strict_exchange_count": 0, "exchange_gain_min": 0.0,
                    "exchange_risk_before": 0.0, "exchange_risk_after": 0.0,
                    "selected_positions": committed, "remasked_positions": [],
                    "base_positions": committed, "baseline_positions": committed,
                    "official_baseline_positions": committed,
                })
                return x
            _, predicted = native_module.sample_tokens(logits[0, start:], temperature=0.0)
            result, row = select(logits[0, start:], before[0, start:], predicted, baseline,
                        mask_id=self.mask, step=i, steps=steps, dacs=dacs, global_offset=start)
            x[0, start:] = result
            trace.append(row)
            return x
        out = self.model.diffusion_generate(prompt, max_new_tokens=length, steps=steps,
                temperature=0.1, top_p=0.9, alg="entropy", alg_temp=0.0,
                return_dict_in_generate=True,
                generation_logits_hook_func=logits_hook, generation_tokens_hook_func=tokens_hook)
        return out.sequences[0, start:], trace

    def llada2_fixed(self, prompt, steps, length, dacs):
        start = prompt.shape[1]
        total = ((start + length + 31) // 32) * 32
        block_starts = list(range(start // 32 * 32, total, 32))
        sizes = [min(b + 32, total) - max(b, start) for b in block_starts]
        if steps < len(sizes):
            raise ValueError("NFE budget cannot cover every native attention block")
        # Allocate all requested forward calls in advance, independent of logits.
        allocation = np.ones(len(sizes), dtype=int)
        for _ in range(steps - len(sizes)):
            priority = np.asarray(sizes) / allocation
            priority[allocation >= sizes] = -1
            allocation[int(np.argmax(priority))] += 1
        x = torch.full((1, total), self.mask, device=prompt.device, dtype=torch.long)
        x[:, :start] = prompt
        pos = torch.arange(total, device=prompt.device)
        attention = ((pos[:, None] // 32) >= (pos[None, :] // 32))[None, None].to(torch.bfloat16)
        trace = []
        for b, count_steps, size in zip(block_starts, allocation, sizes):
            end = min(b + 32, total)
            mutable = max(b, start)
            for quota in schedule(size, int(count_steps)):
                logits = self.model(x[:, :end], attention_mask=attention[:, :, :end, :end],
                                    position_ids=pos[:end][None]).logits[0, mutable:end].clone()
                # Fixed-length controlled M2T decoding cannot execute structural edits.
                for name in ("delete_token_id", "split_token_id"):
                    token = getattr(self.model.config, name, None)
                    if token is not None:
                        logits[:, token] = -torch.inf
                logits[:, self.mask] = -torch.inf
                predicted = logits.argmax(-1)
                confidence = logits.float().softmax(-1).gather(-1, predicted[:, None]).squeeze(-1)
                baseline = confidence.masked_fill(x[0, mutable:end] != self.mask, -torch.inf).topk(quota).indices
                result, row = select(logits, x[0, mutable:end], predicted, baseline, mask_id=self.mask,
                           step=len(trace), steps=steps, dacs=dacs, global_offset=mutable)
                x[0, mutable:end] = result
                row.update(block_start=b, block_length=32, native_t2t_edits=False, native_structural_edits=False)
                trace.append(row)
        return x[0, start:start + length], trace

    def llada_fast(self, prompt, steps, length, dacs, adaptive, guard=False):
        start, blocks = prompt.shape[1], length // 32
        if length % 32 or steps % blocks:
            raise ValueError("Fast-dLLM fixed budget must divide blocks")
        local_steps = 32 if adaptive else steps // blocks
        x = torch.full((1, start + length), self.mask, device=prompt.device, dtype=torch.long)
        x[:, :start] = prompt
        trace = []
        # Match the pinned Fast-dLLM NFE convention: count one explicit model
        # call per scheduled forward. Remote-code wrappers can make the global
        # forward monkey-patch counter differ by one on long threshold runs.
        explicit_nfe = 0
        for block in range(blocks):
            s, e = start + block * 32, start + (block + 1) * 32
            cache = None
            replace = torch.zeros_like(x, dtype=torch.bool)
            replace[:, s:e] = True
            quotas = schedule(32, local_steps)
            for i in range(local_steps):
                if i == 0:
                    output = self.model(x, use_cache=True)
                    explicit_nfe += 1
                    cache = output.past_key_values
                    logits = output.logits[0, s:e]
                else:
                    logits = self.model(x[:, s:e], past_key_values=cache, use_cache=True,
                                        replace_position=replace).logits[0]
                    explicit_nfe += 1
                if adaptive and not bool((x[0, s:e] == self.mask).any()):
                    raise RuntimeError("unexpected redundant adaptive forward")
                current = x[0, s:e]
                predicted = logits.argmax(-1)
                guarded = logits.to(torch.float64)
                # Official Fast-dLLM does not add an EOS confidence suppressor.
                # Keep the optional guard only for explicit diagnostics; normal
                # fast/fixed and fast/threshold jobs must match the pinned
                # Fast-dLLM sampler exactly.
                if guard:
                    guarded[:, 126081] = -torch.inf
                confidence = guarded.softmax(-1).gather(-1, predicted[:, None]).squeeze(-1)
                confidence = confidence.masked_fill(current != self.mask, -torch.inf)
                if adaptive:
                    chosen = (confidence >= 0.9) & (current == self.mask)
                    chosen[confidence.argmax()] = True
                    baseline = chosen.nonzero().flatten()
                else:
                    # Upstream Fast-dLLM uses sort, not topk, to break ties.
                    if i == 0:
                        full = torch.full((x.shape[1],), torch.finfo(confidence.dtype).min,
                                          device=x.device, dtype=confidence.dtype)
                        full[s:e] = confidence.masked_fill(current != self.mask, torch.finfo(confidence.dtype).min)
                        baseline = torch.sort(full, descending=True).indices[:quotas[i]] - s
                    else:
                        baseline = torch.sort(confidence, descending=True).indices[:quotas[i]]
                result, row = select(logits, current, predicted, baseline, mask_id=self.mask,
                    step=(int((x[0, start:] != self.mask).sum()) if adaptive else len(trace)),
                    steps=length if adaptive else steps, dacs=dacs, suppress=(126081,) if guard else (), global_offset=s)
                x[0, s:e] = result
                row.update(block=block, cache="dual", threshold=0.9 if adaptive else None)
                trace.append(row)
                if adaptive and not bool((result == self.mask).any()):
                    break
        self.nfe = explicit_nfe
        return x[0, start:], trace

    def dream_fast(self, prompt, steps, length, dacs, adaptive):
        start, blocks = prompt.shape[1], length // 32
        if length % 32 or steps % blocks:
            raise ValueError("Fast-dLLM fixed budget must divide blocks")
        local_steps = steps // blocks
        x = torch.full((1, start + length), self.mask, device=prompt.device, dtype=torch.long)
        x[:, :start] = prompt
        trace = []
        sample = self.upstream_fast.sample_tokens
        times = torch.linspace(1, 1e-3, local_steps + 1, device=x.device)
        explicit_nfe = 0
        for block in range(blocks):
            s, e = start + block * 32, start + (block + 1) * 32
            first = self.model(x, "full", None, use_cache=True)
            explicit_nfe += 1
            cache = first.past_key_values
            shifted = torch.cat([first.logits[:, :1], first.logits[:, :-1]], dim=1)[0, s:e]
            _, predicted = sample(shifted, temperature=0.0)
            # Upstream Dream warms the cache and commits the block's first token.
            x[0, s:e], row = select(shifted, x[0, s:e], predicted,
                    torch.tensor([0], device=x.device), mask_id=self.mask,
                    step=len(trace), steps=steps, dacs=False, global_offset=s)
            row.update(block=block, cache="dual", prefill=True)
            trace.append(row)
            replace = torch.zeros_like(x, dtype=torch.bool)
            replace[:, s:e] = True
            for i in range(1, 33 if adaptive else local_steps):
                out = self.model(x[:, s:e], "full", None, past_key_values=cache,
                                 use_cache=True, dual_cache=True, replace_position=replace)
                explicit_nfe += 1
                logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)[0]
                current = x[0, s:e]
                confidence, predicted = sample(logits, temperature=0.0, neg_entropy=not adaptive)
                confidence = confidence.masked_fill(current != self.mask, -torch.inf)
                masks = int((current == self.mask).sum())
                if adaptive:
                    chosen = (confidence >= 0.9) & (current == self.mask)
                    chosen[confidence.argmax()] = True
                    baseline = chosen.nonzero().flatten()
                else:
                    quota = int(masks * (1 - times[i + 1] / times[i])) if i < local_steps - 1 else masks
                    baseline = confidence.topk(quota).indices
                x[0, s:e], row = select(logits, current, predicted, baseline,
                        mask_id=self.mask, step=(int((x[0, start:] != self.mask).sum()) if adaptive else len(trace)),
                        steps=length if adaptive else steps, dacs=dacs, global_offset=s)
                row.update(block=block, cache="dual", threshold=0.9 if adaptive else None)
                trace.append(row)
                if adaptive and not bool((x[0, s:e] == self.mask).any()):
                    break
        # Remote-code wrappers can invoke the underlying forward more than
        # once for a cache update.  Fast-dLLM's protocol counts scheduled
        # generation calls, so use the explicit call counter for audits and
        # equivalence checks rather than the wrapper's incidental count.
        self.nfe = explicit_nfe
        return x[0, start:], trace
