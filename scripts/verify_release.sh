#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m py_compile \
  "$ROOT/setcommit/policy.py" "$ROOT/setcommit/selector.py" \
  "$ROOT/setcommit/eb.py" "$ROOT/setcommit/adapters.py" \
  "$ROOT/examples/run_smoke.py"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/verify_manifest.py"
PYTHON_BIN="$PYTHON_BIN" bash "$ROOT/scripts/run_smoke.sh"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$ROOT/examples/test_torch_remask.py"
test -s "$ROOT/examples/smoke_result.json"
echo "release verification passed"
