"""Policies for the dependency-aware DACS experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DependencyAwareConfig:
    name: str = "dacs_base"
    lambda_start: float = 0.35
    lambda_end: float = 0.10
    tau_start: float = 16.0
    tau_end: float = 4.0
    use_dependency_risk: bool = True
    risk_budget_scale: float | None = 1.0
    exchange: bool = True
    exchange_ratio: float = 0.20
    exchange_window_start: float = 0.0
    exchange_window_end: float = 0.8
    exchange_top_k: int = 2
    min_posterior_gain: float = 0.0
    structure_protection: bool = True
    structure_penalty: float = 0.10
    # Conservative controls keep dependency scheduling as a tie-breaker for
    # high-confidence positions instead of allowing it to replace top-k.
    reliability: str = "margin"
    shortlist_multiplier: float = 1.0
    confidence_floor: float = 0.0
    min_topk_overlap: float = 0.0
    max_penalty_fraction: float = 1.0
    adaptive_schedule: bool = False
    schedule: str = "uniform"

    def __post_init__(self) -> None:
        if self.lambda_start < 0 or self.lambda_end < 0:
            raise ValueError("lambda values must be non-negative")
        if self.tau_start <= 0 or self.tau_end <= 0:
            raise ValueError("tau values must be positive")
        if self.risk_budget_scale is not None and self.risk_budget_scale < 0:
            raise ValueError("risk_budget_scale must be non-negative")
        if not 0 <= self.exchange_ratio <= 1:
            raise ValueError("exchange_ratio must be in [0, 1]")
        if not 0 <= self.exchange_window_start <= self.exchange_window_end <= 1:
            raise ValueError("invalid exchange window")
        if self.exchange_top_k < 1:
            raise ValueError("exchange_top_k must be positive")
        if self.min_posterior_gain < 0:
            raise ValueError("min_posterior_gain must be non-negative")
        if self.structure_penalty < 0:
            raise ValueError("structure_penalty must be non-negative")
        if self.reliability not in {"margin", "posterior"}:
            raise ValueError("reliability must be margin or posterior")
        if self.shortlist_multiplier < 1:
            raise ValueError("shortlist_multiplier must be at least one")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be in [0, 1]")
        if not 0 <= self.min_topk_overlap <= 1:
            raise ValueError("min_topk_overlap must be in [0, 1]")
        if self.max_penalty_fraction < 0:
            raise ValueError("max_penalty_fraction must be non-negative")
        if self.schedule != "uniform":
            raise ValueError("only uniform schedule is implemented")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def config_from_dict(raw: dict[str, object]) -> DependencyAwareConfig:
    values = dict(raw)
    return DependencyAwareConfig(**values)
