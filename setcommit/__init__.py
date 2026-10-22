"""SETCOMMIT: Coordinated Token Commitments.

The public API is deliberately small: an upstream masked-diffusion host
supplies logits, the current state and its transfer quota; :func:`select`
returns an auditable set of positions with the same net quota.  No model
weights, training code or task scorer are required by this package.
"""

from .policy import (
    Config,
    NATIVE_FAST,
    EB_QUALITY_GUARD,
    DACS_V5_EXCHANGE_020,  # compatibility name for the internal experiment id
)
from .selector import apply_selection, select_positions
from .eb import entropy_bound_quota, entropy_bound_baseline
from .adapters import select_after_upstream

__all__ = [
    "Config", "NATIVE_FAST", "EB_QUALITY_GUARD",
    "DACS_V5_EXCHANGE_020", "select_positions", "apply_selection",
    "entropy_bound_quota", "entropy_bound_baseline", "select_after_upstream",
]

__version__ = "1.0.0"
