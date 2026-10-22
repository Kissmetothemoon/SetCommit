"""Fixed-transfer DACS with actual risk selection and equal-cardinality repair.

Only generation logits are promoted to FP32. Greedy selection uses compact
CPU arrays to avoid a CUDA synchronization for every candidate decision.
The posterior control uses the official full-sequence torch.topk, including
its tie behavior. No training, verifier or extra model forward is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import torch

from dacs_method_experiment.sampler import uniform_schedule
from vendor.LLaDA.generate import add_gumbel_noise


def risk_matrix(uncertainty: np.ndarray, density: float, tau: float) -> np.ndarray:
    positions = np.arange(len(uncertainty))
    risk = density * np.exp(-np.abs(positions[:, None] - positions[None, :]) / tau)
    risk *= uncertainty[:, None] * uncertainty[None, :]
    np.fill_diagonal(risk, 0.0)
    return risk


def set_risk(positions: np.ndarray, risk: np.ndarray) -> float:
    return float(risk[np.ix_(positions, positions)].sum() / 2.0)


def choose(candidates: np.ndarray, reliability: np.ndarray, confidence: np.ndarray,
           predicted: np.ndarray, risk: np.ndarray, baseline: np.ndarray,
           *, count: int, progress: float, config: Any,
           structure_token_ids: set[int]) -> tuple[np.ndarray, dict]:
    """Greedy order, bounded posterior deviation, fixed cardinality.

    A confidence floor is soft: exhaust allowed candidates first, then relax
    it to finish the official budget. The overlap constraint is enforced by
    reserving enough remaining slots for the posterior baseline set.
    """
    count = min(count, len(candidates))
    if count == 0:
        return candidates[:0], {"risk": 0.0, "budget": None, "floor_relaxations": 0}
    if len(baseline) != count or not set(baseline).issubset(candidates):
        raise ValueError("baseline must contain exactly count candidate positions")
    required = math.ceil(config.min_topk_overlap * count)
    # This also preserves torch.topk's original treatment of tied scores.
    if required == count:
        return baseline.copy(), {"risk": set_risk(baseline, risk), "budget": None,
                                 "floor_relaxations": 0}
    values = reliability[candidates].astype(np.float64)
    finite = np.isfinite(values)
    q = np.full(len(candidates), -np.inf)
    if finite.any():
        low, high = values[finite].min(), values[finite].max()
        q[finite] = (values[finite] - low) / (high - low) if high - low > 1e-8 else 1.0
    else:
        q[:] = 0.0
    order = np.lexsort((candidates, -q))
    size = min(len(candidates), max(count, math.ceil(count * config.shortlist_multiplier)))
    # Include the posterior baseline so the overlap constraint is feasible.
    take = np.isin(candidates, candidates[order[:size]]) | np.isin(candidates, baseline)
    positions, q = candidates[take], q[take]
    pair = risk[np.ix_(positions, positions)]
    baseline_mask = np.isin(positions, baseline)
    anchors = np.isin(predicted[positions], list(structure_token_ids))
    allowed = confidence[positions] >= config.confidence_floor
    budget = None
    if config.use_dependency_risk and config.risk_budget_scale is not None and count >= 2:
        all_pairs = max(len(candidates) * (len(candidates) - 1) / 2, 1)
        budget = (set_risk(candidates, risk) / all_pairs * count * (count - 1) / 2
                  * config.risk_budget_scale * (0.5 + progress))
    chosen = np.zeros(len(positions), dtype=bool)
    inc = np.zeros(len(positions))
    selected: list[int] = []
    current_risk, baseline_hits, relaxations = 0.0, 0, 0
    lam = config.lambda_start + (config.lambda_end - config.lambda_start) * progress
    for slot in range(count):
        eligible = ~chosen
        if required - baseline_hits >= count - slot:
            eligible &= baseline_mask
        if (eligible & allowed).any():
            eligible &= allowed
        else:
            relaxations += int(config.confidence_floor > 0)
        penalty = np.minimum(lam * inc, config.max_penalty_fraction) if config.use_dependency_risk else 0.0
        gain = q - penalty
        if config.structure_protection:
            # Soft anchor protection; confident independent anchors may coexist.
            gain = gain - config.structure_penalty * anchors
        if budget is not None:
            feasible = eligible & (current_risk + inc <= budget + 1e-8)
            if feasible.any():
                eligible = feasible
        gain = np.where(eligible, gain, -np.inf)
        if np.isfinite(gain).any():
            best = int(np.argmax(gain))
        else:
            remaining = np.flatnonzero(eligible)
            if not len(remaining):
                raise RuntimeError("selector cannot preserve cardinality")
            best = int(remaining[0])
        chosen[best] = True
        selected.append(int(positions[best]))
        baseline_hits += int(baseline_mask[best])
        current_risk += float(inc[best])
        inc += pair[:, best]
    return np.asarray(selected, dtype=np.int64), {
        "risk": current_risk, "budget": budget, "floor_relaxations": relaxations}


def repair(base: np.ndarray, candidates: np.ndarray, committed: np.ndarray,
           current: np.ndarray, predicted: np.ndarray, old_p: np.ndarray,
           new_p: np.ndarray, top_ids: np.ndarray, risk: np.ndarray,
           select_extra: Any, *, step: int, steps: int, config: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    """Swap only additional commits against old commits, keeping base intact."""
    empty = np.asarray([], dtype=np.int64)
    info = {"eligible_exchange_count": 0, "exchange_old_posterior_sum": 0.0,
            "exchange_new_posterior_sum": 0.0, "exchange_gain_sum": 0.0,
            "exchange_gain_min": 0.0, "exchange_risk_before": 0.0,
            "exchange_risk_after": 0.0, "strict_exchange_count": 0,
            "exchange_cap": 0}
    progress = (step + 1) / steps
    if not config.exchange or not config.exchange_window_start <= progress <= config.exchange_window_end:
        return base, empty, info
    cap = min(len(candidates) - len(base), math.floor(config.exchange_ratio * len(base)))
    info["exchange_cap"] = cap
    if cap <= 0:
        return base, empty, info
    eligible = committed[(current[committed] != predicted[committed])
                         & ~(top_ids[committed, :config.exchange_top_k] == current[committed, None]).any(-1)
                         & (new_p[committed] > old_p[committed] + config.min_posterior_gain)]
    eligible = eligible[np.lexsort((eligible, old_p[eligible]))]
    info["eligible_exchange_count"] = len(eligible)
    # Compare equal-size post-step sets, with and without exchange.
    before = set_risk(np.concatenate((committed, base)), risk)
    for amount in range(min(cap, len(eligible)), 0, -1):
        extra = select_extra(amount)
        if len(extra) != amount or np.isin(extra, base).any():
            raise RuntimeError("exchange extension must preserve the base set")
        remasked = eligible[:amount]
        gains = np.sort(new_p[extra]) - np.sort(old_p[remasked])
        if not (gains > config.min_posterior_gain).all():
            continue
        kept = committed[~np.isin(committed, remasked)]
        selected = np.concatenate((base, extra))
        after = set_risk(np.concatenate((kept, selected)), risk)
        if after > before + 1e-7:
            continue
        info.update(exchange_old_posterior_sum=float(old_p[remasked].sum()),
                    exchange_new_posterior_sum=float(new_p[extra].sum()),
                    exchange_gain_sum=float(gains.sum()), exchange_gain_min=float(gains.min()),
                    exchange_risk_before=before, exchange_risk_after=after,
                    strict_exchange_count=amount)
        return selected, remasked, info
    return base, empty, info


@dataclass
class Trace:
    policy: dict
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def generate(model: Any, prompt: torch.Tensor, attention_mask: torch.Tensor | None = None,
             *, config: Any, steps: int = 512, gen_length: int = 512,
             block_length: int = 512, temperature: float = 0.0,
             cfg_scale: float = 0.0, mask_id: int = 126336,
             structure_token_ids: set[int] | None = None,
             confidence_suppress_tokens: tuple[int, ...] = (),
             dense_telemetry: bool = False) -> tuple[torch.Tensor, list[Trace]]:
    if gen_length != block_length or not 0 < steps <= gen_length or cfg_scale or dense_telemetry:
        raise ValueError("v5 requires one block, 1..length steps, no CFG/dense telemetry")
    if prompt.shape[0] != 1:
        raise ValueError("v5 audited protocol requires batch size 1")
    start = prompt.shape[1]
    x = torch.full((1, start + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :start] = prompt
    if attention_mask is not None:
        attention_mask = torch.cat((attention_mask, torch.ones_like(x[:, start:], dtype=attention_mask.dtype)), -1)
    trace = Trace(config.to_dict())
    for step, count in enumerate(uniform_schedule(gen_length, steps)):
        logits = model(x, attention_mask=attention_mask).logits[:, start:]
        predicted_gpu = add_gumbel_noise(logits, temperature).argmax(-1)[0]
        fp32 = logits[0].float()
        raw = fp32.softmax(-1)
        new_p_gpu = raw.gather(-1, predicted_gpu[:, None]).squeeze(-1)
        current_gpu = x[0, start:]
        old_p_gpu = raw.gather(-1, current_gpu[:, None]).squeeze(-1)
        top = fp32.topk(max(2, config.exchange_top_k), -1)
        margin_gpu = top.values[:, 0] - top.values[:, 1]
        guarded = fp32.clone() if confidence_suppress_tokens else fp32
        for token in confidence_suppress_tokens:
            if 0 <= token < guarded.shape[-1]:
                guarded[:, token] = -torch.inf
                margin_gpu = margin_gpu.masked_fill(predicted_gpu == token, -torch.inf)
        probabilities = guarded.softmax(-1)
        confidence_gpu = probabilities.gather(-1, predicted_gpu[:, None]).squeeze(-1)
        entropy_gpu = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
        entropy_gpu /= math.log(probabilities.shape[-1])
        signals = torch.stack((margin_gpu, confidence_gpu, entropy_gpu, old_p_gpu, new_p_gpu)).cpu().numpy()
        margin, confidence, uncertainty, old_p, new_p = signals
        current = current_gpu.cpu().numpy().copy()
        predicted = predicted_gpu.cpu().numpy()
        top_ids = top.indices.cpu().numpy()
        candidates = np.flatnonzero(current == mask_id)
        committed = np.flatnonzero(current != mask_id)
        progress = (step + 1) / steps
        tau = config.tau_start + (config.tau_end - config.tau_start) * progress
        risk = risk_matrix(np.clip(uncertainty, 0, 1), len(candidates) / gen_length, tau)
        # Match official top-k over the FULL sequence (tie handling included).
        full_conf = torch.full((x.shape[1],), -torch.inf, device=x.device)
        full_conf[start:] = confidence_gpu.masked_fill(current_gpu != mask_id, -torch.inf)
        baseline = (full_conf.topk(count).indices - start).cpu().numpy()
        reliability = confidence if config.reliability == "posterior" else margin
        base, detail = choose(candidates, reliability, confidence, predicted, risk, baseline,
                              count=count, progress=progress, config=config,
                              structure_token_ids=structure_token_ids or set())

        def select_extra(amount: int) -> np.ndarray:
            remaining = candidates[~np.isin(candidates, base)]
            extra_conf = full_conf.clone()
            extra_conf[torch.as_tensor(base + start, device=x.device)] = -torch.inf
            extra_baseline = (extra_conf.topk(amount).indices - start).cpu().numpy()
            return choose(remaining, reliability, confidence, predicted, risk, extra_baseline,
                          count=amount, progress=progress, config=config,
                          structure_token_ids=structure_token_ids or set())[0]

        selected, remasked, info = repair(base, candidates, committed, current, predicted,
                                         old_p, new_p, top_ids, risk, select_extra,
                                         step=step, steps=steps, config=config)
        if len(np.unique(selected)) != len(selected) or not np.isin(selected, candidates).all():
            raise RuntimeError("invalid commit set")
        if len(selected) - len(remasked) != count:
            raise RuntimeError("net-commit budget changed")
        idx = torch.as_tensor(selected, device=x.device)
        x[0, torch.as_tensor(remasked + start, device=x.device)] = mask_id
        x[0, idx + start] = predicted_gpu[idx]
        gaps = np.diff(np.sort(selected))
        trace.steps.append({"net_commits": count, "new_commits": len(selected),
            "remasked": len(remasked), "exchange_count": len(remasked),
            "mask_count": len(candidates), "candidate_count": len(candidates),
            "selected_count": len(selected), "selected_adjacent_pairs": int((gaps == 1).sum()),
            "selected_min_gap": int(gaps.min()) if len(gaps) else 0,
            "selected_mean_gap": float(gaps.mean()) if len(gaps) else 0.0,
            "selected_margin_mean": float(np.nan_to_num(margin[selected], neginf=0).mean()),
            "selected_entropy_mean": float(uncertainty[selected].mean()),
            "risk_before": set_risk(baseline, risk), "risk_after": set_risk(base, risk),
            "risk_budget": detail["budget"], "floor_relaxations": detail["floor_relaxations"],
            "lambda_t": config.lambda_start + (config.lambda_end - config.lambda_start) * progress,
            "tau_t": tau, "baseline_overlap_count": int(np.isin(base, baseline).sum()),
            "base_positions": base.tolist(), "selected_positions": selected.tolist(),
            "remasked_positions": remasked.tolist(), "baseline_positions": baseline.tolist(), **info})
        del logits, fp32, guarded, raw, probabilities, top
    if bool((x[:, start:] == mask_id).any()):
        raise RuntimeError("unresolved masks")
    return x, [trace]
