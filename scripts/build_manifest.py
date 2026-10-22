#!/usr/bin/env python3
"""Generate the deterministic release manifest used by reviewers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {"__pycache__", ".cache", ".git", ".hg", ".svn", ".venv", ".pytest_cache"}
SKIP_NAMES = {"RELEASE_MANIFEST.json"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name in SKIP_NAMES:
        continue
    if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
        continue
    rel = path.relative_to(ROOT).as_posix()
    files.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)})

manifest = {
    "schema": 2,
    "release": "setcommit",
    "method": "SETCOMMIT: Coordinated Token Commitments",
    "paper": "iclr2027_conference.pdf",
    "version": "1.0.0",
    "legacy_config_mapping": {
        "dacs_v5_exchange_020": "NATIVE_FAST",
        "dacs_v5_quality_guard": "EB_QUALITY_GUARD",
    },
    "model_assets": {
        "repo": "GSAI-ML/LLaDA-8B-Instruct",
        "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
    },
    "smoke": {
        "command": "PYTHON_BIN=<python-with-numpy> bash scripts/verify_release.sh",
        "result": "examples/smoke_result.json",
    },
    "files": files,
}
(ROOT / "RELEASE_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"wrote {len(files)} files")
