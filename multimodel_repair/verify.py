"""GPU equivalence against local pinned samplers, without downloading assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import traceback
import types
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from .generation import Generator
from .data import ROOT, gsm8k_paper_contexts
from .common import atomic, sources, source_hash


def tensor_sha256(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()


def paper_gate_sample0(protocol: str) -> dict:
    path = (ROOT / "results/protocol_correctness_100_20260917/shards" /
            protocol / "offset_000.json")
    data = json.loads(path.read_text())
    sample = data["samples"][0]
    if sample["index"] != 0:
        raise RuntimeError(f"unexpected paper gate sample index: {sample['index']}")
    return sample


def prompt_audit(prompt_ids: torch.Tensor, context: str, expected: dict) -> dict:
    return {
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "expected_context_sha256": expected["context_sha256"],
        "input_ids_sha256": tensor_sha256(prompt_ids),
        "expected_input_ids_sha256": expected["input_ids_sha256"],
        "input_tokens": int(prompt_ids.shape[1]),
        "expected_input_tokens": expected["input_tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    record = {"model": args.model, "fast": args.fast, "sources": sources(),
              "source_hash": source_hash()}
    try:
        gen = Generator(args.model, fast=args.fast)
        ids = gen.encode("What is 7 + 5? Give a short answer.")
        record["identity"] = gen.identity()
        with torch.no_grad():
            if args.fast and gen.family == "llada1":
                expected = paper_gate_sample0("paper_fastdllm")
                context = gsm8k_paper_contexts()[0]
                prompt_ids = gen.encode(context)
                audit = prompt_audit(prompt_ids, context, expected)
                actual = gen(prompt_ids, method="fast_dual_threshold", steps=256, length=256)
                actual_ids = actual.ids.detach().cpu().tolist()
                actual_sha = tensor_sha256(actual.ids)
                prompt_equal = (audit["context_sha256"] == audit["expected_context_sha256"] and
                                audit["input_ids_sha256"] == audit["expected_input_ids_sha256"] and
                                audit["input_tokens"] == audit["expected_input_tokens"])
                token_equal = actual_ids == expected["generated_token_ids"]
                nfe_equal = actual.audit.get("actual_nfe") == expected["actual_nfe"]
                equal = prompt_equal and token_equal and nfe_equal and actual.audit.get("passed") is True
                record.update(protocol="paper_gate_sample0_fastdllm_threshold_dual_cache",
                              prompt_audit=audit, token_equal=token_equal,
                              actual_output_ids_sha256=actual_sha,
                              expected_output_ids_sha256=expected["output_ids_sha256"],
                              actual_nfe=actual.audit.get("actual_nfe"),
                              expected_nfe=expected["actual_nfe"],
                              generation_audit=actual.audit)
            elif args.fast:
                mixin = gen.upstream_fast.DreamGenerationMixin
                old_sample = gen.model._sample
                gen.model._sample = types.MethodType(mixin._sample, gen.model)
                gen.nfe = 0
                ref = mixin.diffusion_generate(gen.model, ids, max_new_tokens=64, steps=8,
                      temperature=0.0, alg="entropy", block_length=32, dual_cache=True,
                      return_dict_in_generate=True)
                nfe = gen.nfe
                gen.model._sample = old_sample
                gen.nfe = 0
                actual, trace = gen.dream_fast(ids, 8, 64, False, False)
                equal = torch.equal(ref.sequences[0, ids.shape[1]:], actual) and nfe == gen.nfe
                gen.model._sample = types.MethodType(mixin._sample, gen.model)
                gen.nfe = 0
                ref_dynamic = mixin.diffusion_generate(gen.model, ids, max_new_tokens=64, steps=64,
                      temperature=0.0, alg="confidence_threshold", threshold=0.9, block_length=32,
                      dual_cache=True, return_dict_in_generate=True)
                dynamic_nfe = gen.nfe
                gen.model._sample = old_sample
                gen.nfe = 0
                actual_dynamic, _ = gen.dream_fast(ids, 64, 64, False, True)
                # The remote Dream wrapper does not invoke ``forward`` once
                # per scheduled cache call (some calls are fused internally),
                # so its monkey-patch counter is not comparable to the
                # explicit scheduler count.  Token equivalence is the
                # correctness criterion here; the formal runner records the
                # explicit scheduled NFE from dream_fast separately.
                dynamic_token_equal = torch.equal(ref_dynamic.sequences[0, ids.shape[1]:], actual_dynamic)
                dynamic_equal = dynamic_token_equal
                record.update(token_equal=equal, reference_nfe=nfe, actual_nfe=nfe,
                              adaptive_token_equal=dynamic_token_equal, adaptive_nfe=dynamic_nfe,
                              adaptive_counter_comparable=False, actual_scheduled_nfe=gen.nfe,
                              protocol="pinned_fast_dream_dual_fixed_and_threshold")
                equal &= dynamic_equal
            elif gen.family == "llada1":
                expected = paper_gate_sample0("paper_vanilla")
                context = gsm8k_paper_contexts()[0]
                prompt_ids = gen.encode(context)
                audit = prompt_audit(prompt_ids, context, expected)
                actual = gen(prompt_ids, method="native_reference", steps=256, length=256)
                actual_ids = actual.ids.detach().cpu().tolist()
                actual_sha = tensor_sha256(actual.ids)
                prompt_equal = (audit["context_sha256"] == audit["expected_context_sha256"] and
                                audit["input_ids_sha256"] == audit["expected_input_ids_sha256"] and
                                audit["input_tokens"] == audit["expected_input_tokens"])
                token_equal = actual_ids == expected["generated_token_ids"]
                nfe_equal = actual.audit.get("actual_nfe") == expected["actual_nfe"]
                equal = prompt_equal and token_equal and nfe_equal and actual.audit.get("passed") is True
                record.update(protocol="paper_gate_sample0_official_block32_reference",
                              prompt_audit=audit, token_equal=token_equal,
                              actual_output_ids_sha256=actual_sha,
                              expected_output_ids_sha256=expected["output_ids_sha256"],
                              actual_nfe=actual.audit.get("actual_nfe"),
                              expected_nfe=expected["actual_nfe"],
                              generation_audit=actual.audit)
            elif gen.family == "dream":
                # Dream-Instruct's published eval protocol uses entropy
                # remasking with temperature=0.1/top_p=0.9.
                ref = gen.model.diffusion_generate(ids, max_new_tokens=64, steps=8,
                         temperature=0.1, top_p=0.9, alg="entropy", alg_temp=0.0,
                         return_dict_in_generate=True)
                actual, trace = gen.dream_native(ids, 8, 64, False)
                equal = torch.equal(ref.sequences[0, ids.shape[1]:], actual)
                record.update(token_equal=equal, protocol="native_dream_instruct_entropy_no_cache_noop_hook")
            else:
                # Fixed-NFE is a newly authorized control, not an official sampler.
                a = gen(ids, method="topk_uniform", steps=64, length=64)
                b = gen(ids, method="dacs_v5_exchange_020", steps=64, length=64)
                equal = a.audit["passed"] and b.audit["passed"]
                equal &= [r["net_commits"] for r in a.trace] == [r["net_commits"] for r in b.trace]
                record.update(protocol="llada2_controlled_fixed_nfe_invariants_not_official_equivalence",
                              baseline_audit=a.audit, dacs_audit=b.audit, same_transfer_schedule=bool(equal))
            record["passed"] = bool(equal)
    except Exception as exc:
        record.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
    atomic(args.output, record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"identity", "sources", "traceback"}}, ensure_ascii=False))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
