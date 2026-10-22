"""NumPy implementation of SETCOMMIT (Coordinated Token Commitments).

The model adapter supplies one logits matrix, the current masked sequence,
the top-1 proposal and the upstream baseline positions. SETCOMMIT returns a new
position set with the same cardinality and an auditable trace. No extra model
forward, training, verifier or task-specific signal is used.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .policy import Config, NATIVE_FAST


def risk_matrix(uncertainty: np.ndarray, density: float, tau: float) -> np.ndarray:
    pos = np.arange(len(uncertainty))
    risk = density * np.exp(-np.abs(pos[:, None] - pos[None, :]) / tau)
    risk *= uncertainty[:, None] * uncertainty[None, :]
    np.fill_diagonal(risk, 0.0)
    return risk


def set_risk(positions: np.ndarray, risk: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(risk[np.ix_(positions, positions)].sum() / 2.0)


def _choose(candidates, reliability, confidence, predicted, risk, baseline,
            *, count, progress, config: Config):
    count = min(int(count), len(candidates))
    if count == 0:
        return np.empty(0, dtype=np.int64), {"risk": 0.0, "floor_relaxations": 0}
    if len(baseline) != count or not np.isin(baseline, candidates).all():
        raise ValueError("baseline must contain exactly count candidates")
    required = math.ceil(config.min_topk_overlap * count)
    values = reliability[candidates].astype(np.float64)
    finite = np.isfinite(values)
    q = np.full(len(candidates), -np.inf)
    if finite.any():
        lo, hi = values[finite].min(), values[finite].max()
        q[finite] = 1.0 if hi - lo <= 1e-8 else (values[finite] - lo) / (hi - lo)
    else:
        q[:] = 0.0
    order = np.lexsort((candidates, -q))
    size = min(len(candidates), max(count, math.ceil(config.shortlist_multiplier * count)))
    keep = np.isin(candidates, candidates[order[:size]]) | np.isin(candidates, baseline)
    positions, q = candidates[keep], q[keep]
    pair = risk[np.ix_(positions, positions)]
    baseline_mask = np.isin(positions, baseline)
    anchors = np.isin(predicted[positions], np.asarray(config.structure_token_ids, dtype=np.int64))
    allowed = confidence[positions] >= config.confidence_floor
    chosen = np.zeros(len(positions), dtype=bool)
    inc = np.zeros(len(positions))
    selected, hits, current_risk, relaxations = [], 0, 0.0, 0
    lam = config.lambda_start + (config.lambda_end - config.lambda_start) * progress
    for slot in range(count):
        eligible = ~chosen
        if required - hits >= count - slot:
            eligible &= baseline_mask
        if (eligible & allowed).any():
            eligible &= allowed
        else:
            relaxations += int(config.confidence_floor > 0)
        penalty = np.minimum(lam * inc, config.max_penalty_fraction) if config.use_dependency_risk else 0.0
        gain = q - penalty
        if config.structure_protection and len(config.structure_token_ids):
            gain = gain - config.structure_penalty * anchors
        gain = np.where(eligible, gain, -np.inf)
        if not np.isfinite(gain).any():
            raise RuntimeError("selector cannot preserve cardinality")
        best = int(np.argmax(gain))
        chosen[best] = True
        selected.append(int(positions[best]))
        hits += int(baseline_mask[best])
        current_risk += float(inc[best])
        inc += pair[:, best]
    return np.asarray(selected, dtype=np.int64), {
        "risk": current_risk, "floor_relaxations": relaxations,
    }


def _repair(base, candidates, committed, current, predicted, old_p, new_p,
            top_ids, risk, extra: Callable[[int], np.ndarray], *, step, steps,
            config: Config):
    empty = np.empty(0, dtype=np.int64)
    info = {"exchange_cap": 0, "eligible_exchange_count": 0,
            "exchange_gain_min": 0.0, "exchange_risk_before": 0.0,
            "exchange_risk_after": 0.0, "strict_exchange_count": 0}
    progress = (step + 1) / steps
    if not config.exchange or not config.exchange_window_start <= progress <= config.exchange_window_end:
        return base, empty, info
    cap = min(len(candidates) - len(base), math.floor(config.exchange_ratio * len(base)))
    info["exchange_cap"] = cap
    if cap <= 0:
        return base, empty, info
    top_old = (top_ids[committed, :config.exchange_top_k] == current[committed, None]).any(-1)
    eligible = committed[(current[committed] != predicted[committed]) & ~top_old
                         & (new_p[committed] > old_p[committed] + config.min_posterior_gain)]
    eligible = eligible[np.argsort(old_p[eligible], kind="stable")]
    info["eligible_exchange_count"] = len(eligible)
    before = set_risk(np.concatenate((committed, base)), risk)
    for amount in range(min(cap, len(eligible)), 0, -1):
        extra_pos = extra(amount)
        if len(extra_pos) != amount or np.isin(extra_pos, base).any():
            raise RuntimeError("extra set overlaps base")
        remasked = eligible[:amount]
        gains = np.sort(new_p[extra_pos]) - np.sort(old_p[remasked])
        if not np.all(gains > config.min_posterior_gain):
            continue
        kept = committed[~np.isin(committed, remasked)]
        after = set_risk(np.concatenate((kept, base, extra_pos)), risk)
        if after > before + 1e-7:
            continue
        info.update(exchange_gain_min=float(gains.min()), exchange_risk_before=before,
                    exchange_risk_after=after, strict_exchange_count=amount)
        return np.concatenate((base, extra_pos)), remasked, info
    return base, empty, info


def select_positions(logits: np.ndarray, current: np.ndarray, predicted: np.ndarray,
                     baseline: np.ndarray, *, mask_id: int = -1, step: int = 0,
                     steps: int = 1, config: Config = NATIVE_FAST,
                     dacs: bool = True,
                     structure_token_ids: tuple[int, ...] | None = None,
                     editable_positions: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Select one transfer set from one model forward.

    ``dacs`` is a compatibility keyword for paired campaign scripts; set it
    to ``False`` to return the host baseline unchanged.

    ``current`` is a mutable generation block unless ``editable_positions`` is
    supplied. The latter is mandatory when prompt/immutable tokens are present
    in the same tensor.
    """
    logits = np.asarray(logits, dtype=np.float64)
    current = np.asarray(current, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != len(current):
        raise ValueError("logits must have shape [sequence, vocabulary]")
    if editable_positions is None:
        # An all-mask tensor is unambiguous. Once visible tokens exist, an
        # explicit editable list is mandatory so prompt tokens cannot be
        # mistaken for previously committed generation tokens.
        if np.any(current != mask_id):
            raise ValueError("editable_positions is required when current contains visible tokens")
        editable = np.arange(len(current), dtype=np.int64)
        input_scope = "generation_block"
    else:
        editable = np.asarray(editable_positions, dtype=np.int64).reshape(-1)
        if len(np.unique(editable)) != len(editable) or np.any(editable < 0) or np.any(editable >= len(current)):
            raise ValueError("editable_positions must be unique in-range indices")
        input_scope = "explicit_editable_positions"
    candidates = editable[current[editable] == mask_id]
    if structure_token_ids is not None:
        # Keep the immutable policy object reusable while allowing a tokenizer
        # to provide its punctuation/control-token IDs at runtime.
        from dataclasses import replace
        config = replace(config, structure_token_ids=tuple(int(x) for x in structure_token_ids))
    count = len(baseline)
    if len(np.unique(baseline)) != len(baseline):
        raise ValueError("baseline positions must be unique")
    if count > len(candidates) or not np.isin(baseline, candidates).all():
        raise ValueError("baseline must contain only masked candidate positions")
    if count == 0 or not dacs:
        return baseline.copy(), {"net_commits": count, "new_commits": count,
            "exchange_count": 0, "exchange_cap": 0,
            "selected_positions": baseline.tolist(), "remasked_positions": [],
            "baseline_positions": baseline.tolist(), "baseline_overlap_count": count,
            "baseline_retention_required": int(math.ceil(config.min_topk_overlap * count)),
            "baseline_retention_passed": True,
            "input_scope": input_scope,
            "risk_before": 0.0, "risk_after": 0.0}
    stable = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(stable); p /= p.sum(axis=1, keepdims=True)
    top_ids = np.argsort(-logits, axis=1, kind="stable")
    confidence = p[np.arange(len(current)), predicted]
    old_idx = np.clip(current, 0, p.shape[1] - 1)
    old_p = p[np.arange(len(current)), old_idx]
    new_p = confidence.copy()
    margin = logits[np.arange(len(current)), top_ids[:, 0]] - logits[np.arange(len(current)), top_ids[:, 1]]
    entropy = -(p * np.log(np.maximum(p, np.finfo(np.float64).tiny))).sum(axis=1) / math.log(p.shape[1])
    # The host baseline is authoritative. In particular, EB supplies an
    # entropy-bound prefix rather than a posterior top-k set.
    full_baseline = baseline.copy()
    progress = (step + 1) / steps
    tau = config.tau_start + (config.tau_end - config.tau_start) * progress
    risk = risk_matrix(np.clip(entropy, 0, 1), len(candidates) / len(current), tau)
    reliability = confidence if config.reliability == "posterior" else margin
    base, detail = _choose(candidates, reliability, confidence, predicted, risk,
                           full_baseline, count=count, progress=progress, config=config)

    def extra(amount: int) -> np.ndarray:
        remaining = candidates[~np.isin(candidates, base)]
        extra_base = remaining[np.argsort(-confidence[remaining], kind="stable")[:amount]]
        return _choose(remaining, reliability, confidence, predicted, risk, extra_base,
                       count=amount, progress=progress, config=config)[0]

    committed = editable[current[editable] != mask_id]
    selected, remasked, info = _repair(base, candidates, committed, current, predicted,
                                      old_p, new_p, top_ids, risk, extra,
                                      step=step, steps=steps, config=config)
    if len(selected) - len(remasked) != count:
        raise RuntimeError("net transfer invariant failed")
    trace = {"net_commits": count, "new_commits": len(selected),
             "exchange_count": len(remasked), "selected_positions": selected.tolist(),
             "remasked_positions": remasked.tolist(), "base_positions": base.tolist(),
             "baseline_positions": full_baseline.tolist(),
             "input_scope": input_scope,
             "baseline_overlap_count": int(np.isin(base, full_baseline).sum()),
             "baseline_retention_required": int(math.ceil(config.min_topk_overlap * count)),
             "baseline_retention_passed": int(np.isin(base, full_baseline).sum()) >= math.ceil(config.min_topk_overlap * count),
             "risk_before": set_risk(full_baseline, risk), "risk_after": set_risk(base, risk),
             "tau_t": tau, "lambda_t": config.lambda_start + (config.lambda_end - config.lambda_start) * progress,
             "floor_relaxations": detail["floor_relaxations"], **info}
    return selected, trace


def apply_selection(current: np.ndarray, predicted: np.ndarray, selected: np.ndarray,
                    trace: dict, *, mask_id: int = -1) -> np.ndarray:
    """Apply remask first and commit the selected top-1 proposals."""
    result = np.asarray(current, dtype=np.int64).copy()
    result[np.asarray(trace["remasked_positions"], dtype=np.int64)] = mask_id
    result[selected] = np.asarray(predicted, dtype=np.int64)[selected]
    if int((result == mask_id).sum()) != int((current == mask_id).sum()) - trace["net_commits"]:
        raise RuntimeError("mask/cardinality invariant failed")
    return result
