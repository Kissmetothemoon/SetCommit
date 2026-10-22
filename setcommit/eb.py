"""Entropy-bounded (EB) host adapter used in the paper's composition study.

EB chooses a quota before SETCOMMIT is called.  The selector may change the
set while preserving that quota; it is therefore important to log both the
original EB bound and the selected-set bound.  These helpers are pure NumPy
and do not perform a model forward.
"""
from __future__ import annotations

import numpy as np


def entropy_bound_quota(error_proxy: np.ndarray, gamma: float) -> int:
    """Return the longest prefix satisfying the EB cumulative bound.

    ``error_proxy`` must already be ordered from easiest to hardest (the EB
    host supplies this order).  The bound is ``sum(prefix) - max(prefix) <=
    gamma``.  The one-token prefix is always admissible for nonnegative gamma.
    """
    values = np.asarray(error_proxy, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0
    if not np.isfinite(values).all():
        raise ValueError("error_proxy must be finite")
    gamma = float(gamma)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    total = 0.0
    largest = -np.inf
    quota = 0
    for i, value in enumerate(values, start=1):
        total += float(value)
        largest = max(largest, float(value))
        if total - largest <= gamma + 1e-12:
            quota = i
        else:
            break
    return quota


def entropy_bound_baseline(masked_positions: np.ndarray, error_proxy: np.ndarray,
                           gamma: float) -> tuple[np.ndarray, dict]:
    """Build the EB pre-prefix and return an auditable quota trace."""
    positions = np.asarray(masked_positions, dtype=np.int64).reshape(-1)
    errors = np.asarray(error_proxy, dtype=np.float64).reshape(-1)
    if len(positions) != len(errors):
        raise ValueError("positions and error_proxy must have equal length")
    order = np.argsort(errors, kind="stable")
    ordered_positions = positions[order]
    ordered_errors = errors[order]
    quota = entropy_bound_quota(ordered_errors, gamma)
    selected = ordered_positions[:quota]
    bound = float(ordered_errors[:quota].sum() - (ordered_errors[:quota].max()
                                                   if quota else 0.0))
    return selected, {
        "gamma": float(gamma), "quota": int(quota),
        "ordered_positions": ordered_positions.tolist(),
        "ordered_error_proxy": ordered_errors.tolist(),
        "original_bound": bound,
    }
