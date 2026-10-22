#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-llada_8b_instruct}
TASK=${TASK:-gsm8k_cot}
KIND=${KIND:-fixed}
STEPS=${STEPS:-64}
LIMIT=${LIMIT:-2}
GPU=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
mkdir -p "$ROOT/results/reviewer_case"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m multimodel_repair.run_pair \
  --model "$MODEL" --task "$TASK" --kind "$KIND" --steps "$STEPS" \
  --limit "$LIMIT" --seed 0 --warmup 1 --timing-repeats 1 \
  --output "$ROOT/results/reviewer_case/${MODEL}_${TASK}_${KIND}_${STEPS}.json"
