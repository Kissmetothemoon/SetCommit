"""Fast-dLLM dual-cache adapter with the DACS-v5 commit-position selector.

The wrapper preserves Fast-dLLM's cache and transfer schedule.  It replaces
only the position selection returned by ``get_transfer_index`` and restores
the original function after the call, so the official baseline remains
untouched and can be run in the same process for paired comparisons.
"""
from __future__ import annotations

import torch

from generate import add_gumbel_noise, generate_with_dual_cache
import generate as _fast


def _dacs_transfer(logits, temperature, remasking, mask_index, x,
                   num_transfer_tokens, threshold=None):
    if threshold is not None:
        return _fast._ORIGINAL_GET_TRANSFER_INDEX(
            logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold
        )
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided for fixed-quota DACS selection")

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
        confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -torch.inf))

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)

    chosen = torch.zeros_like(mask_index)
    for row in range(logits.shape[0]):
        positions = torch.nonzero(mask_index[row], as_tuple=False).flatten()
        budget = min(int(num_transfer_tokens[row].item()), int(positions.numel()))
        if budget <= 0:
            continue
        uncertainty = (1.0 - confidence[row].clamp_min(0.0)).to(torch.float64)
        remaining = positions.tolist()
        selected: list[int] = []
        while remaining and len(selected) < budget:
            best, best_score = remaining[0], None
            for pos in remaining:
                risk = confidence.new_zeros(())
                for prev in selected:
                    risk = risk + torch.exp(confidence.new_tensor(-abs(pos - prev) / 4.0)) * uncertainty[pos] * uncertainty[prev]
                score = confidence[row, pos] - 0.2 * risk
                if best_score is None or bool(score > best_score):
                    best, best_score = pos, score
            selected.append(best)
            remaining.remove(best)
        chosen[row, torch.tensor(selected, device=logits.device)] = True
    return x0, chosen & mask_index


def generate_fast_dllm_dacs_v5(*args, **kwargs):
    """Run Fast-dLLM dual cache with DACS-v5 exchange-ratio=0.20 selector."""
    raise RuntimeError("Retired local wrapper lacked exchange: use multimodel_repair.generation; upstream generate.py is unchanged")
    if hasattr(_fast, "_ORIGINAL_GET_TRANSFER_INDEX"):
        raise RuntimeError("nested DACS selector patch")
    _fast._ORIGINAL_GET_TRANSFER_INDEX = _fast.get_transfer_index
    _fast.get_transfer_index = _dacs_transfer
    try:
        return generate_with_dual_cache(*args, **kwargs)
    finally:
        _fast.get_transfer_index = _fast._ORIGINAL_GET_TRANSFER_INDEX
        delattr(_fast, "_ORIGINAL_GET_TRANSFER_INDEX")
