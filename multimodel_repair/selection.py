"""Shared frozen v5 choose/repair, with bulk CPU transfer and auditable budgets."""
from __future__ import annotations

import math
import os
import numpy as np
import torch

from dacs_method_v5.policy import load_policy
from dacs_method_v5.sampler import choose, repair, risk_matrix, set_risk

# A campaign may select a frozen policy per process without changing the
# baseline/sampler implementation.  The default remains the historical v5
# configuration so existing experiments are reproducible.
POLICY_NAME = os.environ.get("DACS_POLICY", "dacs_v5_exchange_020")
POLICY = load_policy(POLICY_NAME)


def select(logits, current, predicted, baseline, *, mask_id, step, steps,
           dacs, suppress=(), protected=(), global_offset=0):
    """Inputs cover one mutable generation block; prompt/cached blocks are excluded.

    baseline contains the exact upstream-selected positions, preserving its
    cardinality (including threshold batches). No per-candidate CUDA operations.
    """
    current_np = current.detach().cpu().numpy().copy()
    baseline_np = baseline.detach().cpu().numpy().astype(np.int64)
    official_baseline_np = baseline_np.copy()
    candidates = np.flatnonzero(current_np == mask_id)
    count = len(baseline_np)
    if count > len(candidates) or not np.isin(baseline_np, candidates).all():
        raise RuntimeError("baseline chose non-mask positions")
    selected, remasked = baseline_np, np.asarray([], dtype=np.int64)
    info = {"exchange_cap": 0, "strict_exchange_count": 0,
            "exchange_gain_min": 0.0, "exchange_risk_before": 0.0, "exchange_risk_after": 0.0}
    base = baseline_np
    if dacs and count:
        fp32 = logits.float()
        raw = fp32.softmax(-1)
        old_p = raw.gather(-1, current[:, None]).squeeze(-1)
        new_p = raw.gather(-1, predicted[:, None]).squeeze(-1)
        top = fp32.topk(max(2, POLICY.exchange_top_k), -1)
        margin = top.values[:, 0] - top.values[:, 1]
        guarded = fp32.clone() if suppress else fp32
        for token in suppress:
            guarded[:, token] = -torch.inf
            margin = margin.masked_fill(predicted == token, -torch.inf)
        p = guarded.softmax(-1) if suppress else raw
        confidence = p.gather(-1, predicted[:, None]).squeeze(-1)
        # V5's 75% overlap is with posterior top-k, even on a backbone whose
        # native control ranks by entropy. Upstream supplies only the quota.
        baseline_np = confidence.masked_fill(current != mask_id, -torch.inf).topk(count).indices.cpu().numpy()
        entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1) / math.log(p.shape[-1])
        margin, confidence, entropy, old_p, new_p = torch.stack((margin, confidence, entropy, old_p, new_p)).cpu().numpy()
        pred = predicted.cpu().numpy()
        top_ids = top.indices.cpu().numpy()
        progress = (step + 1) / steps
        tau = POLICY.tau_start + (POLICY.tau_end - POLICY.tau_start) * progress
        risk = risk_matrix(np.clip(entropy, 0, 1), len(candidates) / len(current_np), tau)
        reliability = confidence if POLICY.reliability == "posterior" else margin
        base, detail = choose(candidates, reliability, confidence, pred, risk, baseline_np,
                              count=count, progress=progress, config=POLICY, structure_token_ids=set())

        def extra(amount):
            remaining = candidates[~np.isin(candidates, base)]
            # Same full-block torch.topk tie behavior as the original v5.
            scores = torch.as_tensor(confidence.copy(), device=logits.device)
            allowed = np.isin(np.arange(len(current_np)), remaining)
            scores.masked_fill_(~torch.as_tensor(allowed, device=logits.device), -torch.inf)
            extra_base = scores.topk(amount).indices.cpu().numpy()
            return choose(remaining, reliability, confidence, pred, risk, extra_base,
                          count=amount, progress=progress, config=POLICY, structure_token_ids=set())[0]

        committed = np.flatnonzero(current_np != mask_id)
        if protected:
            committed = committed[~np.isin(committed, protected)]
        selected, remasked, info = repair(base, candidates, committed, current_np, pred,
                old_p, new_p, top_ids, risk, extra, step=step, steps=steps, config=POLICY)
        info.update(risk_before=set_risk(baseline_np, risk), risk_after=set_risk(base, risk),
                    baseline_overlap_count=int(np.isin(base, baseline_np).sum()), **detail)
    if len(selected) - len(remasked) != count or np.intersect1d(selected, remasked).size:
        raise RuntimeError("v5 net transfer invariant failed")
    if not np.isin(selected, candidates).all() or np.isin(remasked, candidates).any():
        raise RuntimeError("invalid v5 commit/remask set")
    result = current.clone()
    sel = torch.as_tensor(selected, device=current.device)
    result[torch.as_tensor(remasked, device=current.device)] = mask_id
    result[sel] = predicted[sel]
    after_masks = int((result == mask_id).sum().item())
    if len(candidates) - after_masks != count:
        raise RuntimeError("proposal includes unresolved mask tokens or changed transfer budget")
    trace = {"step": step, "offset": global_offset, "mask_count": len(candidates),
             "mask_after": after_masks, "net_commits": count, "new_commits": len(selected),
             "exchange_count": len(remasked), "selected_positions": selected.tolist(),
             "remasked_positions": remasked.tolist(), "base_positions": base.tolist(),
             "baseline_positions": baseline_np.tolist(),
             "official_baseline_positions": official_baseline_np.tolist(), **info}
    return result, trace


def audit_trace(trace: list[dict], *, expected_nfe: int | None, actual_nfe: int) -> dict:
    errors = []
    if expected_nfe is not None and actual_nfe != expected_nfe:
        errors.append(f"NFE {actual_nfe} != {expected_nfe}")
    if expected_nfe is not None and len(trace) != actual_nfe:
        errors.append("trace/forward count mismatch")
    for row in trace:
        if row["new_commits"] - row["exchange_count"] != row["net_commits"]:
            errors.append("net transfer mismatch")
        if row["exchange_count"] > row.get("exchange_cap", 0):
            errors.append("exchange cap exceeded")
        if "selected_positions" in row:
            selected = row["selected_positions"]
            remasked = row["remasked_positions"]
            if len(set(selected)) != len(selected) or set(selected) & set(remasked):
                errors.append("invalid commit/remask positions")
            if len(selected) != row["new_commits"] or len(remasked) != row["exchange_count"]:
                errors.append("trace position/count mismatch")
        if row["exchange_count"]:
            if row["exchange_gain_min"] <= POLICY.min_posterior_gain:
                errors.append("strict posterior gain failed")
            if row["exchange_risk_after"] > row["exchange_risk_before"] + 1e-7:
                errors.append("exchange increased risk")
    return {"passed": not errors, "errors": errors, "actual_nfe": actual_nfe,
            "expected_nfe": expected_nfe, "trace_steps": len(trace),
            "exchange_count": sum(r["exchange_count"] for r in trace),
            "net_commits": sum(r["net_commits"] for r in trace)}
