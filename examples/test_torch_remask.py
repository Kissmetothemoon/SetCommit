#!/usr/bin/env python3
"""CPU regression for the actual torch Native selection bridge."""
from __future__ import annotations

import torch

from multimodel_repair.selection import select


def main() -> int:
    length, vocab, mask = 10, 8, 7
    logits = torch.zeros(length, vocab)
    logits[:, 0] = 20
    logits[0, 2], logits[0, 0] = 20, 0  # visible old token 1 is stale
    current = torch.full((length,), mask, dtype=torch.long)
    current[0] = 1
    predicted = logits.argmax(-1)
    baseline = torch.arange(1, 6)
    output, trace = select(logits, current, predicted, baseline,
                           mask_id=mask, step=0, steps=4, dacs=True)
    passed = (trace["exchange_count"] >= 1
              and 0 in trace["remasked_positions"]
              and int(output[0]) == mask
              and trace["new_commits"] - trace["exchange_count"] == 5)
    print("torch_native_remask=", passed,
          "remasked=", trace["remasked_positions"],
          "exchange=", trace["exchange_count"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
