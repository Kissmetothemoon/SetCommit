"""Dependency-aware commit scheduling sampler described in ``method.md``.

The sampler performs one model forward per diffusion step.  Position selection
uses normalized top-1/top-2 logit margin, normalized entropy, an exponentially
decaying dependency kernel, a progress-dependent risk penalty, and a soft
risk-budget feasibility check.  A finite posterior-dominance exchange is
applied after selection without changing the official net commit budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StepTrace:
    net_commits: int
    new_commits: int
    remasked: int
    exchange_count: int
    mask_count: int
    selected_adjacent_pairs: int = 0
    selected_min_gap: int = 0
    selected_mean_gap: float = 0.0
    selected_margin_mean: float = 0.0
    selected_entropy_mean: float = 0.0
    risk_before: float = 0.0
    risk_after: float = 0.0
    risk_budget: float | None = None
    lambda_t: float = 0.0
    tau_t: float = 0.0
    candidate_count: int = 0
    selected_count: int = 0
    eligible_exchange_count: int = 0
    exchange_old_posterior_sum: float = 0.0
    exchange_new_posterior_sum: float = 0.0
    exchange_gain_sum: float = 0.0
    exchange_gain_min: float = 0.0
    exchange_risk_before: float = 0.0
    exchange_risk_after: float = 0.0
    strict_exchange_count: int = 0


@dataclass
class GenerationTrace:
    policy: dict[str, object]
    steps: list[StepTrace] = field(default_factory=list)
    dense_steps: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "steps": [asdict(x) for x in self.steps]}


def uniform_schedule(total: int, steps: int) -> list[int]:
    if total < 0 or steps < 1:
        raise ValueError("total must be non-negative and steps positive")
    return [total // steps + int(i < total % steps) for i in range(steps)]


def _normalize(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    selected = values[positions].float()
    if selected.numel() == 0:
        return torch.zeros_like(values, dtype=torch.float32)
    out = torch.zeros_like(values, dtype=torch.float32)
    finite = torch.isfinite(selected)
    if not bool(finite.any()):
        # Preserve a deterministic fallback when every candidate is guarded
        # (normally impossible for a non-empty language-model block).
        out[positions] = 0.0
        return out
    finite_values = selected[finite]
    low, high = finite_values.min(), finite_values.max()
    finite_positions = positions[finite]
    if float(high - low) > 1e-8:
        out[finite_positions] = (finite_values - low) / (high - low)
    else:
        out[finite_positions] = 1.0
    # Guarded predictions remain ineligible unless all candidates are guarded.
    out[positions[~finite]] = -torch.inf
    return out


def _normalize_candidates(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Normalize only the selected candidate slice.

    The previous implementation materialized a full sequence-sized tensor for
    every selector call.  DACS only consumes candidate scores, so keeping the
    compact slice avoids repeated indexing and allocations in the greedy loop.
    """
    selected = values[positions].float()
    if selected.numel() == 0:
        return selected
    finite = torch.isfinite(selected)
    out = torch.full_like(selected, -torch.inf, dtype=torch.float32)
    if not bool(finite.any()):
        out.zero_()
        return out
    finite_values = selected[finite]
    low, high = finite_values.min(), finite_values.max()
    if float(high - low) > 1e-8:
        out[finite] = (finite_values - low) / (high - low)
    else:
        out[finite] = 1.0
    return out


def _risk_components(positions: torch.Tensor, uncertainty: torch.Tensor, *,
                     density: float, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.numel() < 2:
        empty = torch.zeros(0, device=positions.device, dtype=torch.float32)
        return empty, empty
    left = positions[:, None]
    right = positions[None, :]
    upper = torch.triu(torch.ones((positions.numel(), positions.numel()),
                                  device=positions.device, dtype=torch.bool), diagonal=1)
    distance = (left - right).abs().float()[upper]
    kernel = float(density) * torch.exp(-distance / max(float(tau), 1e-6))
    values = kernel * uncertainty[positions[:, None].expand_as(upper)][upper] * uncertainty[positions[None, :].expand_as(upper)][upper]
    return values, distance


def _risk_matrix(positions: torch.Tensor, uncertainty: torch.Tensor, *,
                 density: float, tau: float) -> torch.Tensor:
    """Return the symmetric pair-risk matrix used by selection and exchange."""
    count = int(positions.numel())
    if count == 0:
        return torch.zeros((0, 0), device=positions.device, dtype=torch.float32)
    distance = (positions[:, None] - positions[None, :]).abs().float()
    risk = (float(density) * torch.exp(-distance / max(float(tau), 1e-6))
            * uncertainty[positions, None] * uncertainty[None, positions])
    # A position cannot be its own dependency partner.
    return risk.fill_diagonal_(0.0)


def set_risk(positions: torch.Tensor, uncertainty: torch.Tensor, *,
             density: float, tau: float) -> float:
    values, _ = _risk_components(positions, uncertainty, density=density, tau=tau)
    return float(values.sum().item()) if values.numel() else 0.0


def _risk_budget(candidates: torch.Tensor, uncertainty: torch.Tensor, count: int,
                 *, density: float, tau: float, progress: float,
                 scale: float | None) -> float | None:
    if scale is None or candidates.numel() < 2 or count < 2:
        return None
    pair_values, _ = _risk_components(candidates, uncertainty, density=density, tau=tau)
    if pair_values.numel() == 0:
        return None
    pair_count = max(count * (count - 1) / 2, 1)
    # The budget follows the method.md early-protection/late-relaxation rule.
    return float(pair_values.mean().item() * pair_count * float(scale) * (0.5 + progress))


def select_dependency_aware(candidates: torch.Tensor, margin: torch.Tensor,
                            uncertainty: torch.Tensor, predicted: torch.Tensor,
                            *, count: int, progress: float, lambda_t: float,
                            tau: float, density: float,
                            risk_budget_scale: float | None,
                            use_dependency_risk: bool,
                            structure_protection: bool,
                            structure_token_ids: set[int],
                            structure_penalty: float,
                            confidence: torch.Tensor | None = None,
                            reliability: str = "margin",
                            shortlist_multiplier: float = 1.0,
                            confidence_floor: float = 0.0,
                            min_topk_overlap: float = 0.0,
                            max_penalty_fraction: float = 1.0) -> tuple[torch.Tensor, float, float | None]:
    """Select exactly ``count`` candidates with bounded risk-aware scoring."""
    count = min(max(int(count), 0), int(candidates.numel()))
    if count == 0:
        return candidates[:0], 0.0, None
    # Posterior-first policies use DACS only as a tie-breaker among a compact
    # high-confidence shortlist. Defaults retain the historical margin mode.
    reliability_values = confidence if reliability == "posterior" and confidence is not None else margin
    q = _normalize_candidates(reliability_values, candidates)
    positions = candidates
    shortlist_size = min(positions.numel(), max(count, int(np.ceil(count * shortlist_multiplier))))
    if shortlist_size < positions.numel():
        order = torch.topk(q, k=shortlist_size, largest=True).indices
        positions, q = positions[order], q[order]
    pair_risk = (_risk_matrix(positions, uncertainty, density=density, tau=tau)
                 if use_dependency_risk else None)
    if risk_budget_scale is not None and use_dependency_risk and positions.numel() >= 2 and count >= 2:
        pair_count = max(count * (count - 1) / 2, 1)
        all_pairs = positions.numel() * (positions.numel() - 1) / 2
        budget = float(pair_risk.triu(diagonal=1).sum().item() / max(all_pairs, 1)
                       * pair_count * float(risk_budget_scale) * (0.5 + progress))
    else:
        budget = None
    selected_mask = torch.zeros(positions.numel(), device=positions.device, dtype=torch.bool)
    current_risk = torch.zeros((), device=positions.device, dtype=torch.float32)
    anchor_count = 0
    if structure_protection and structure_token_ids:
        anchor_mask = torch.zeros_like(selected_mask)
        predicted_candidates = predicted[positions]
        for token_id in structure_token_ids:
            anchor_mask |= predicted_candidates == int(token_id)
    else:
        anchor_mask = torch.zeros_like(selected_mask)
    for _ in range(count):
        if bool(selected_mask.all()):
            break
        if use_dependency_risk and bool(selected_mask.any()):
            inc = pair_risk[:, selected_mask].sum(-1)
        else:
            inc = torch.zeros(positions.numel(), device=positions.device, dtype=torch.float32)
        if use_dependency_risk and max_penalty_fraction >= 0 and bool(torch.isfinite(q).any()):
            finite_q = q[torch.isfinite(q)]
            q_range = (finite_q.max() - finite_q.min()).clamp_min(1e-6)
            penalty = (float(lambda_t) * inc).clamp(max=float(max_penalty_fraction) * q_range)
            gain = q - penalty
        else:
            gain = q - float(lambda_t) * inc if use_dependency_risk else q.clone()
        gain = gain.masked_fill(selected_mask, -torch.inf)
        if structure_protection and structure_penalty:
            # Penalize anchors softly while the hard rule below prevents a
            # second anchor from consuming the same transfer slice.
            gain = gain - float(structure_penalty) * anchor_mask.to(gain.dtype)
        if structure_protection and anchor_count >= 1:
            if bool((~anchor_mask & ~selected_mask).any()):
                gain = gain.masked_fill(anchor_mask, -torch.inf)
        feasible = torch.ones_like(gain, dtype=torch.bool)
        if budget is not None:
            feasible = current_risk + inc <= float(budget) + 1e-8
            if bool(feasible.any()):
                gain = gain.masked_fill(~feasible, -torch.inf)
        if not torch.isfinite(gain).any():
            # Preserve the fixed transfer cardinality if the risk budget is
            # too strict or every remaining candidate is a structure anchor.
            gain = (q - float(lambda_t) * inc if use_dependency_risk else q.clone())
            gain = gain.masked_fill(selected_mask, -torch.inf)
        finite_gain = torch.isfinite(gain) & ~selected_mask
        if bool(finite_gain.any()):
            best = int(torch.argmax(gain).item())
        else:
            # Guarded EOS/EOT predictions can leave the final candidate with
            # an all-infinite score.  Select an unchosen position explicitly
            # instead of argmax(-inf), which would repeat an existing index.
            remaining_indices = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            if remaining_indices.numel() == 0:
                break
            best = int(remaining_indices[0].item())
        selected_mask[best] = True
        current_risk += inc[best] if use_dependency_risk else 0.0
        if bool(anchor_mask[best].item()):
            anchor_count += 1
    return positions[selected_mask], float(current_risk.item()), budget


def _repair_exchange(x_row: torch.Tensor, block_positions: torch.Tensor,
                     candidates: torch.Tensor, predicted: torch.Tensor,
                     raw_probabilities: torch.Tensor, top_token_ids: torch.Tensor,
                     select_fn: Callable[[int], tuple[torch.Tensor, float, float | None]],
                     *, net_commits: int, step: int, steps: int,
                     config: Any, uncertainty: torch.Tensor, density: float,
                     tau: float) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float | int]]:
    telemetry: dict[str, float | int] = {
        "eligible_exchange_count": 0, "exchange_old_posterior_sum": 0.0,
        "exchange_new_posterior_sum": 0.0, "exchange_gain_sum": 0.0,
        "exchange_gain_min": 0.0, "exchange_risk_before": 0.0,
        "exchange_risk_after": 0.0, "strict_exchange_count": 0,
    }
    masked_count = int(candidates.sum().item())
    base = min(int(net_commits), masked_count)
    base_selected, _, _ = select_fn(base)
    if not config.exchange or masked_count <= base:
        return base_selected, block_positions[:0], 0, telemetry
    progress = (step + 1) / steps
    if not config.exchange_window_start <= progress <= config.exchange_window_end:
        return base_selected, block_positions[:0], 0, telemetry
    cap = min(masked_count - base, int(config.exchange_ratio * base))
    if cap <= 0:
        return base_selected, block_positions[:0], 0, telemetry
    local_candidates = candidates[block_positions]
    committed = block_positions[~local_candidates]
    eligible: list[tuple[float, int]] = []
    for position in committed.tolist():
        old = int(x_row[position].item()); new = int(predicted[position].item())
        if old == new:
            continue
        old_p = float(raw_probabilities[position, old].item())
        new_p = float(raw_probabilities[position, new].item())
        # ``top_token_ids`` was computed once for the whole block alongside
        # the margin.  Re-running a 126k-vocabulary top-k for every committed
        # position makes the bounded exchange check dominate low-step latency.
        if (not bool((top_token_ids[position, :config.exchange_top_k] == old).any())
                and new_p > old_p + config.min_posterior_gain):
            eligible.append((old_p, position))
    eligible.sort(key=lambda item: item[0])
    telemetry["eligible_exchange_count"] = len(eligible)
    cap = min(cap, len(eligible))
    # The strict matching loop only changes the number of exchanged pairs;
    # the greedy ordering of the expanded candidate set is stable.  Compute
    # that ordering once instead of rerunning the Python selector for every
    # possible cap during the usual fallback descent.
    expanded_selected, _, _ = select_fn(base + cap)
    while cap:
        selected = expanded_selected[:base + cap]
        prospective = selected[base:base + cap]
        new_conf = raw_probabilities[prospective, predicted[prospective]].sort(descending=True).values.tolist()
        matched = sum(new_p > old_p + config.min_posterior_gain
                      for new_p, (old_p, _) in zip(new_conf, eligible[:cap]))
        if matched == cap:
            remasked = torch.tensor([position for _, position in eligible[:cap]], device=x_row.device, dtype=torch.long)
            before = set_risk(committed, uncertainty, density=density, tau=tau)
            kept = committed[~torch.isin(committed, remasked)]
            after_positions = torch.cat((kept, selected), dim=0)
            after = set_risk(after_positions, uncertainty, density=density, tau=tau)
            if after <= before + 1e-7:
                old_values = [old_p for old_p, _ in eligible[:cap]]
                gains = [new_p - old_p for new_p, old_p in zip(new_conf, old_values)]
                telemetry.update({
                    "exchange_old_posterior_sum": float(sum(old_values)),
                    "exchange_new_posterior_sum": float(sum(new_conf)),
                    "exchange_gain_sum": float(sum(gains)),
                    "exchange_gain_min": float(min(gains)) if gains else 0.0,
                    "exchange_risk_before": before, "exchange_risk_after": after,
                    "strict_exchange_count": cap,
                })
                return selected, remasked, cap, telemetry
        cap -= 1
    return base_selected, block_positions[:0], 0, telemetry


@torch.no_grad()
def generate(model: Any, prompt: torch.Tensor, attention_mask: torch.Tensor | None = None,
             *, config: Any, steps: int = 512, gen_length: int = 512,
             block_length: int = 512, temperature: float = 0.0,
             cfg_scale: float = 0.0, mask_id: int = 126336,
             structure_token_ids: set[int] | None = None,
             confidence_suppress_tokens: tuple[int, ...] = (),
             dense_telemetry: bool = False) -> tuple[torch.Tensor, list[GenerationTrace]]:
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("gen_length must divide block_length and steps per block")
    if cfg_scale != 0:
        raise ValueError("dependency-aware sampler uses one unconditional forward")
    try:
        from vendor.LLaDA.generate import add_gumbel_noise
    except ImportError:  # compatibility when the vendored directory is on sys.path
        from generate import add_gumbel_noise

    structure_token_ids = structure_token_ids or set()
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt
    if attention_mask is not None:
        attention_mask = torch.cat((attention_mask, torch.ones((prompt.shape[0], gen_length),
                              dtype=attention_mask.dtype, device=model.device)), dim=-1)
    traces = [GenerationTrace(policy=config.to_dict()) for _ in range(prompt.shape[0])]
    blocks = gen_length // block_length
    local_steps = steps // blocks
    vocab_size = int(getattr(model.config, "vocab_size", 126347))
    log_vocab = float(np.log(max(vocab_size, 2)))
    for block in range(blocks):
        start, end = prompt.shape[1] + block * block_length, prompt.shape[1] + (block + 1) * block_length
        for step, net_commits in enumerate(uniform_schedule(block_length, local_steps)):
            mask_index = x == mask_id
            candidates = mask_index.clone(); candidates[:, :start] = False; candidates[:, end:] = False
            # Keep the model's BF16 logits on device.  The official sampler
            # also computes confidence in model dtype; casting the full
            # 126k-vocabulary tensor to FP32 doubles bandwidth for no
            # material change in the position ordering.
            logits = model(x, attention_mask=attention_mask).logits
            # The protocol's EOS/EOT guard applies to the confidence used for
            # position selection, while predictions remain the raw model argmax
            # exactly as in the repaired official sampler.
            raw_probabilities = F.softmax(logits, dim=-1)
            # Keep confidence separate from raw posterior used by exchange.
            confidence_logits = logits.float()
            for token_id in confidence_suppress_tokens:
                if 0 <= int(token_id) < confidence_logits.shape[-1]:
                    confidence_logits[..., int(token_id)] = -torch.inf
            confidence_probabilities = F.softmax(confidence_logits, dim=-1)
            # Reuse the single softmax for guarded entropy.  For excluded
            # token mass e, H(p | allowed) is computed analytically instead of
            # cloning and renormalizing another full vocabulary tensor.
            probabilities_float = raw_probabilities.float()
            entropy = -(probabilities_float * probabilities_float.clamp_min(
                torch.finfo(probabilities_float.dtype).tiny).log()).sum(-1) / log_vocab
            if confidence_suppress_tokens:
                excluded_mass = torch.zeros_like(entropy)
                excluded_plogp = torch.zeros_like(entropy)
                for token_id in confidence_suppress_tokens:
                    if 0 <= int(token_id) < raw_probabilities.shape[-1]:
                        excluded = probabilities_float[..., int(token_id)]
                        excluded_mass = excluded_mass + excluded
                        excluded_plogp = excluded_plogp + excluded * excluded.clamp_min(
                            torch.finfo(probabilities_float.dtype).tiny).log()
                allowed_mass = (1.0 - excluded_mass).clamp_min(
                    torch.finfo(probabilities_float.dtype).tiny)
                entropy = (entropy * log_vocab + excluded_plogp +
                           allowed_mass * allowed_mass.log()) / allowed_mass / log_vocab
            predicted = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            top_predictions = logits.topk(k=2, dim=-1)
            top_values = top_predictions.values
            margin = top_values[..., 0] - top_values[..., 1]
            if confidence_suppress_tokens:
                suppressed = torch.zeros_like(predicted, dtype=torch.bool)
                for token_id in confidence_suppress_tokens:
                    suppressed |= predicted == int(token_id)
                margin = margin.masked_fill(suppressed, -torch.inf)
            progress = (step + 1) / local_steps
            lambda_t = config.lambda_start + (config.lambda_end - config.lambda_start) * progress
            tau_t = config.tau_start + (config.tau_end - config.tau_start) * progress
            positions = torch.arange(start, end, device=x.device)
            for row in range(x.shape[0]):
                candidate_positions = positions[candidates[row, positions]]
                density = float(candidates[row, start:end].float().mean().item())
                uncertainty = entropy[row].clamp(0, 1)
                # Exchange validation and telemetry request the same base
                # selections more than once in a step.  Memoizing by count
                # preserves exact ordering while avoiding duplicate greedy
                # passes and their CUDA kernel launches.
                selection_cache: dict[int, tuple[torch.Tensor, float, float | None]] = {}

                def select_fn(count: int) -> tuple[torch.Tensor, float, float | None]:
                    key = int(count)
                    if key not in selection_cache:
                        selection_cache[key] = select_dependency_aware(
                            candidate_positions, margin[row], uncertainty, predicted[row], count=count,
                            progress=progress, lambda_t=lambda_t, tau=tau_t, density=density,
                            risk_budget_scale=config.risk_budget_scale,
                            use_dependency_risk=config.use_dependency_risk,
                            structure_protection=config.structure_protection,
                            structure_token_ids=structure_token_ids,
                            structure_penalty=config.structure_penalty,
                            confidence=confidence_probabilities[row].gather(-1, predicted[row].unsqueeze(-1)).squeeze(-1),
                            reliability=config.reliability,
                            shortlist_multiplier=config.shortlist_multiplier,
                            confidence_floor=config.confidence_floor,
                            min_topk_overlap=config.min_topk_overlap,
                            max_penalty_fraction=config.max_penalty_fraction)
                    return selection_cache[key]

                selected, remasked, exchanges, telemetry = _repair_exchange(
                    x[row], positions, candidates[row], predicted[row], raw_probabilities[row],
                    top_predictions.indices[row], select_fn,
                    net_commits=net_commits, step=step, steps=local_steps, config=config,
                    uncertainty=uncertainty, density=density, tau=tau_t)
                selected_probability = raw_probabilities[row, selected, predicted[row, selected]]
                selected_sorted = selected.sort().values
                gaps = selected_sorted[1:] - selected_sorted[:-1]
                adjacent = int((gaps == 1).sum().item()) if gaps.numel() else 0
                min_gap = int(gaps.min().item()) if gaps.numel() else 0
                mean_gap = float(gaps.float().mean().item()) if gaps.numel() else 0.0
                risk_before = (set_risk(positions[~candidates[row, positions]], uncertainty,
                                        density=density, tau=tau_t)
                               if config.use_dependency_risk else 0.0)
                risk_after = (set_risk(selected, uncertainty, density=density, tau=tau_t)
                              if config.use_dependency_risk else 0.0)
                _, _, budget = select_fn(min(int(net_commits), int(candidate_positions.numel())))
                if remasked.numel():
                    x[row, remasked] = mask_id
                    selected = selected[~torch.isin(selected, remasked)]
                x[row, selected] = predicted[row, selected]
                if int(selected.numel()) - int(remasked.numel()) != net_commits:
                    raise RuntimeError(
                        "dependency-aware net-commit invariant violated: "
                        f"selected={selected.numel()} remasked={remasked.numel()} "
                        f"net={net_commits} candidates={candidate_positions.numel()} "
                        f"exchanges={exchanges}")
                traces[row].steps.append(StepTrace(
                    net_commits, int(selected.numel()), int(remasked.numel()), exchanges,
                    int(candidates[row].sum().item()), adjacent, min_gap, mean_gap,
                    float(margin[row, selected].mean().item()) if selected.numel() else 0.0,
                    float(uncertainty[selected].mean().item()) if selected.numel() else 0.0,
                    risk_before, risk_after, budget, lambda_t, tau_t,
                    int(candidate_positions.numel()), int(selected.numel()),
                    int(telemetry["eligible_exchange_count"]),
                    float(telemetry["exchange_old_posterior_sum"]),
                    float(telemetry["exchange_new_posterior_sum"]),
                    float(telemetry["exchange_gain_sum"]), float(telemetry["exchange_gain_min"]),
                    float(telemetry["exchange_risk_before"]), float(telemetry["exchange_risk_after"]),
                    int(telemetry["strict_exchange_count"])))
                if dense_telemetry:
                    traces[row].dense_steps.append({
                        "candidate_positions": (candidate_positions - start).to(torch.int32).cpu().numpy().astype(np.uint16),
                        "candidate_margin_q16": torch.clamp(torch.round(
                            torch.sigmoid(margin[row, candidate_positions].float()) * 65535.0), 0, 65535
                        ).to(torch.int32).cpu().numpy().astype(np.uint16),
                        "candidate_entropy_q16": torch.clamp(torch.round(
                            uncertainty[candidate_positions].float() * 65535.0), 0, 65535
                        ).to(torch.int32).cpu().numpy().astype(np.uint16),
                        "selected_positions": (selected - start).to(torch.int32).cpu().numpy().astype(np.uint16),
                    })
    if bool((x[:, prompt.shape[1]:] == mask_id).any()):
        raise RuntimeError("dependency-aware sampler ended with unresolved masks")
    return x, traces
