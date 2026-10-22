#!/usr/bin/env python3
"""Verify every file hash and size recorded in RELEASE_MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
errors = []
for entry in manifest.get("files", []):
    path = ROOT / entry["path"]
    if not path.is_file():
        errors.append(f"missing: {entry['path']}")
        continue
    actual_size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_size != entry.get("bytes"):
        errors.append(f"size: {entry['path']} ({actual_size} != {entry.get('bytes')})")
    if digest != entry.get("sha256"):
        errors.append(f"sha256: {entry['path']}")
if errors:
    raise SystemExit("manifest verification failed:\n" + "\n".join(errors))
print(f"manifest verification passed ({len(manifest['files'])} files)")
