from dataclasses import dataclass
import json
from pathlib import Path

from dacs_method_experiment.policy import DependencyAwareConfig


@dataclass(frozen=True)
class Config(DependencyAwareConfig):
    name: str = "dacs_v5_risk"
    lambda_start: float = 0.03
    lambda_end: float = 0.01
    risk_budget_scale: float | None = None
    exchange: bool = False
    structure_protection: bool = False
    structure_penalty: float = 0.02
    reliability: str = "margin"
    shortlist_multiplier: float = 4.0
    min_topk_overlap: float = 0.75
    max_penalty_fraction: float = 0.10


def load_policy(name: str) -> Config:
    policies = json.loads(Path(__file__).with_name("policies.json").read_text())
    return Config(name=name, **policies[name])
