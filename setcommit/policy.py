"""Paper Table 3 policies.

``dacs_v5_exchange_020`` is retained as an internal configuration identifier
because it names the frozen Native/Fast campaign.  The public method name is
SETCOMMIT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    name: str
    reliability: str = "margin"
    lambda_start: float = 0.03
    lambda_end: float = 0.01
    tau_start: float = 16.0
    tau_end: float = 4.0
    use_dependency_risk: bool = True
    risk_budget_scale: float | None = None
    shortlist_multiplier: float = 4.0
    min_topk_overlap: float = 0.75
    max_penalty_fraction: float = 0.10
    exchange: bool = True
    exchange_ratio: float = 0.20
    min_posterior_gain: float = 0.002
    exchange_window_start: float = 0.0
    exchange_window_end: float = 0.8
    exchange_top_k: int = 2
    structure_protection: bool = False
    structure_penalty: float = 0.02
    structure_token_ids: tuple[int, ...] = ()
    confidence_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.reliability not in {"margin", "posterior"}:
            raise ValueError("reliability must be margin or posterior")
        if not 0 <= self.exchange_ratio <= 1:
            raise ValueError("exchange_ratio must be in [0, 1]")
        if not 0 <= self.exchange_window_start <= self.exchange_window_end <= 1:
            raise ValueError("invalid exchange window")
        if not 0 <= self.min_topk_overlap <= 1:
            raise ValueError("min_topk_overlap must be in [0, 1]")
        if self.shortlist_multiplier < 1:
            raise ValueError("shortlist_multiplier must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NATIVE_FAST = Config(name="dacs_v5_exchange_020")
EB_QUALITY_GUARD = Config(
    name="dacs_v5_quality_guard",
    reliability="posterior",
    shortlist_multiplier=2.0,
    min_topk_overlap=0.90,
    exchange_ratio=0.10,
    min_posterior_gain=0.005,
    exchange_window_start=0.10,
    exchange_window_end=0.60,
    structure_protection=True,
    structure_penalty=0.10,
)

# Stable compatibility symbol for scripts from the internal campaign.
DACS_V5_EXCHANGE_020 = NATIVE_FAST
