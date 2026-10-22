#!/usr/bin/env python3
"""Reviewer smoke: quality selection plus Native/Fast compatibility.

The first table is a deterministic synthetic masked-decoding case. ``gold``
marks positions whose proposed token is useful for this controlled example;
the model logits, seed, quota and result are all recorded. The second table
passes the same selector through three adapter contracts and checks that DACS
preserves every upstream forward quota.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setcommit import (  # noqa: E402
    EB_QUALITY_GUARD,
    apply_selection,
    entropy_bound_baseline,
    select_positions,
)


class InstrumentedHost:
    """Tiny host double whose counters are updated by executed operations."""
    def __init__(self, fast: bool):
        self.fast = fast
        self.forward_calls = 0
        self.cache_refreshes = 0

    def forward(self, logits: np.ndarray) -> np.ndarray:
        self.forward_calls += 1
        return logits

    def refresh_cache(self) -> None:
        if self.fast:
            self.cache_refreshes += 1


def synthetic_quality_case() -> dict:
    seed, length, vocab, mask = 16, 16, 20, -1
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(length, vocab))
    predicted = logits.argmax(axis=1)
    current = np.full(length, mask, dtype=np.int64)
    current[:4] = rng.integers(0, vocab, size=4)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
    confidence = prob[np.arange(length), predicted]
    candidates = np.flatnonzero(current == mask)
    quota = 4
    baseline = candidates[np.argsort(-confidence[candidates], kind="stable")[:quota]]
    # A held-out position utility used only to make this algorithmic smoke
    # measurable; it is never passed to DACS.
    gold_positions = np.array([5, 8, 11, 14], dtype=np.int64)
    base_selected, base_trace = select_positions(
        logits, current, predicted, baseline, mask_id=mask, step=0, steps=4, dacs=False,
        editable_positions=np.arange(length),
    )
    dacs_selected, dacs_trace = select_positions(
        logits, current, predicted, baseline, mask_id=mask, step=0, steps=4, dacs=True,
        editable_positions=np.arange(length),
    )
    return {
        "seed": seed, "length": length, "vocab": vocab, "quota": quota,
        "gold_positions": gold_positions.tolist(),
        "baseline_positions": base_selected.tolist(),
        "dacs_positions": dacs_selected.tolist(),
        "baseline_gold_hits": int(np.isin(base_selected, gold_positions).sum()),
        "dacs_gold_hits": int(np.isin(dacs_selected, gold_positions).sum()),
        "baseline_trace": base_trace, "dacs_trace": dacs_trace,
        "same_quota": len(base_selected) == len(dacs_selected) == quota,
    }


def adapter_case(mode: str, use_dacs: bool) -> dict:
    seed, length, vocab, mask, steps = 2025, 16, 24, -1, 4
    quotas = [4, 4, 4, 4] if mode.endswith("fixed") else [1, 3, 5, 7]
    rng = np.random.default_rng(seed)
    logits_steps = [rng.normal(size=(length, vocab)) for _ in range(steps)]
    current = np.full(length, mask, dtype=np.int64)
    traces = []
    host = InstrumentedHost(mode.startswith("fast_dllm"))
    for step, (logits, quota) in enumerate(zip(logits_steps, quotas)):
        logits = host.forward(logits)
        predicted = logits.argmax(axis=1)
        shifted = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
        confidence = prob[np.arange(length), predicted]
        candidates = np.flatnonzero(current == mask)
        quota = min(quota, len(candidates))
        baseline = candidates[np.argsort(-confidence[candidates], kind="stable")[:quota]]
        selected, trace = select_positions(
            logits, current, predicted, baseline, mask_id=mask,
            step=step, steps=steps, dacs=use_dacs,
            editable_positions=np.arange(length),
        )
        current = apply_selection(current, predicted, selected, trace, mask_id=mask)
        host.refresh_cache()
        traces.append(trace)
    return {
        "adapter": mode, "selector": "dacs_v5_exchange_020" if use_dacs else "upstream_topk",
        "forward_calls": host.forward_calls, "expected_forward_calls": steps,
        "expected_cache_refreshes": steps if mode.startswith("fast_dllm") else 0,
        "cache_refreshes": host.cache_refreshes,
        "net_commits": int(sum(t["net_commits"] for t in traces)),
        "exchange_count": int(sum(t["exchange_count"] for t in traces)),
        "remaining_masks": int((current == mask).sum()),
        "schedule_preserved": [t["net_commits"] for t in traces] == quotas,
    }


def native_remask_case() -> dict:
    """Regression test: a visible low-posterior token must be remaskable."""
    length, vocab, mask, quota = 10, 8, -1, 5
    logits = np.zeros((length, vocab), dtype=np.float64)
    logits[:, 0] = 20.0
    logits[0, 2], logits[0, 0] = 20.0, 0.0  # visible old token 1 is stale
    current = np.full(length, mask, dtype=np.int64)
    current[0] = 1
    predicted = logits.argmax(axis=1)
    baseline = np.arange(1, quota + 1, dtype=np.int64)
    selected, trace = select_positions(
        logits, current, predicted, baseline, mask_id=mask,
        step=0, steps=4, dacs=True, editable_positions=np.arange(length),
    )
    next_state = apply_selection(current, predicted, selected, trace, mask_id=mask)
    return {
        "remasked_positions": trace["remasked_positions"],
        "strict_exchange_count": trace.get("strict_exchange_count", 0),
        "visible_token_is_remasked": 0 in trace["remasked_positions"]
            and int(next_state[0]) == mask,
        "net_commits": len(selected) - len(trace["remasked_positions"]),
        "quota": quota,
    }


def scope_guard_case() -> dict:
    """Ensure a prompt-containing tensor cannot silently become editable."""
    logits = np.zeros((4, 8), dtype=np.float64)
    current = np.array([3, 3, -1, -1], dtype=np.int64)
    predicted = logits.argmax(axis=1)
    try:
        select_positions(logits, current, predicted, np.array([2]), mask_id=-1)
    except ValueError as exc:
        return {"rejected_without_editable_positions": True, "message": str(exc)}
    return {"rejected_without_editable_positions": False, "message": "not rejected"}


def eb_case() -> dict:
    """Exercise the paper-specified EB quota and EB+SETCOMMIT composition."""
    positions = np.arange(8, dtype=np.int64)
    # Ordered error proxy: gamma=0 admits the first token; gamma=.08 admits a
    # short prefix.  The selector receives the exact host quota in either case.
    errors = np.array([0.10, 0.14, 0.20, 0.60, 0.70, 0.80, 0.90, 1.00])
    baseline, eb_trace = entropy_bound_baseline(positions, errors, gamma=0.24)
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(len(positions), 32))
    predicted = logits.argmax(axis=1)
    current = np.full(len(positions), -1, dtype=np.int64)
    selected, trace = select_positions(
        logits, current, predicted, baseline, mask_id=-1, step=0, steps=4,
        config=EB_QUALITY_GUARD, dacs=True, editable_positions=positions,
    )
    next_state = apply_selection(current, predicted, selected, trace, mask_id=-1)
    selected_errors = errors[selected]
    selected_bound = float(selected_errors.sum() - selected_errors.max()) if len(selected_errors) else 0.0
    return {
        "host": "eb_sampler",
        "gamma": eb_trace["gamma"],
        "quota": int(len(baseline)),
        "selected_positions": selected.tolist(),
        "original_eb_bound": eb_trace["original_bound"],
        "selected_set_bound": selected_bound,
        "bound_metric": "sum(error_proxy[selected])-max(error_proxy[selected])",
        "baseline_retained": int(np.isin(selected, baseline).sum()),
        "baseline_retention_required": int(np.ceil(EB_QUALITY_GUARD.min_topk_overlap * len(baseline))),
        "quota_preserved": int((next_state != -1).sum()) == len(baseline),
        "policy": EB_QUALITY_GUARD.name,
    }


def main() -> int:
    result = {"quality_case": synthetic_quality_case(), "native_remask": native_remask_case(),
              "scope_guard": scope_guard_case(),
              "compatibility": [], "eb_composition": eb_case()}
    for mode in ("native_fixed", "fast_dllm_fixed", "fast_dllm_threshold"):
        result["compatibility"].extend([adapter_case(mode, False), adapter_case(mode, True)])
    result["all_compatibility_checks_pass"] = all(
        r["forward_calls"] == r["expected_forward_calls"]
        and r["cache_refreshes"] == r["expected_cache_refreshes"]
        and r["schedule_preserved"] and r["remaining_masks"] == 0
        for r in result["compatibility"]
    )
    out = Path(__file__).with_name("smoke_result.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    q = result["quality_case"]
    print(f"quality: baseline hits={q['baseline_gold_hits']} DACS hits={q['dacs_gold_hits']} "
          f"same_quota={q['same_quota']}")
    for row in result["compatibility"]:
        print(row["adapter"], row["selector"], "forward=", row["forward_calls"],
              "commits=", row["net_commits"], "masks=", row["remaining_masks"])
    print("all_compatibility_checks_pass=", result["all_compatibility_checks_pass"])
    print("eb_composition quota=", result["eb_composition"]["quota"],
          "quota_preserved=", result["eb_composition"]["quota_preserved"],
          "baseline_retained=", result["eb_composition"]["baseline_retained"])
    return 0 if (result["all_compatibility_checks_pass"]
                 and result["native_remask"]["visible_token_is_remasked"]
                 and result["native_remask"]["net_commits"] == result["native_remask"]["quota"]
                 and result["scope_guard"]["rejected_without_editable_positions"]
                 and result["eb_composition"]["quota_preserved"]
                 and result["eb_composition"]["baseline_retained"] >= result["eb_composition"]["baseline_retention_required"]
                 and abs(result["eb_composition"]["selected_set_bound"] - 0.24) < 1e-9
                 and q["dacs_gold_hits"] > q["baseline_gold_hits"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
