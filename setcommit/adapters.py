"""Backend-neutral adapter contract for Native, Fast-dLLM and EB hosts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .policy import Config, NATIVE_FAST
from .selector import apply_selection, select_positions


class SamplerAdapter(Protocol):
    """Backend contract: SETCOMMIT receives a quota and returns positions."""

    def forward_logits(self, state: np.ndarray) -> np.ndarray: ...
    def upstream_positions(self, logits: np.ndarray, state: np.ndarray) -> np.ndarray: ...
    def refresh_cache(self, changed_positions: np.ndarray) -> None: ...


@dataclass
class AdapterStep:
    forward_calls: int = 0
    cache_refreshes: int = 0
    traces: list[dict] = field(default_factory=list)


def select_after_upstream(logits: np.ndarray, state: np.ndarray, *, mask_id: int,
                          baseline: np.ndarray, step: int, steps: int,
                          use_dacs: bool, config: Config = NATIVE_FAST):
    """One backend-neutral step.

    The host owns forward passes, KV-cache refresh, block boundaries, EOS
    handling and stopping.  SETCOMMIT only receives the host quota and returns
    a same-quota commit/remask edit.  ``use_dacs`` is retained as a concise
    compatibility switch for paired baseline runs.
    """
    predicted = logits.argmax(axis=1)
    selected, trace = select_positions(
        logits, state, predicted, baseline, mask_id=mask_id,
        step=step, steps=steps, config=config, dacs=use_dacs,
        editable_positions=np.arange(len(state), dtype=np.int64),
    )
    next_state = apply_selection(state, predicted, selected, trace, mask_id=mask_id)
    return next_state, trace
