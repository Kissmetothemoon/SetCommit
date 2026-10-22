#!/usr/bin/env python3
"""Download pinned LLaDA assets and record reproducible dataset metadata."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from datasets import load_dataset
from huggingface_hub import snapshot_download


MODEL_REPO = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DATASETS = {
    "gsm8k": {"path": "openai/gsm8k", "config": "main", "split": "test", "revision": "main"},
    "math500": {"path": "HuggingFaceH4/MATH-500", "config": None, "split": "test", "revision": "main"},
    "mbpp": {"path": "google-research-datasets/mbpp", "config": "full", "split": "test", "revision": "main"},
    "humaneval": {"path": "openai/openai_humaneval", "config": None, "split": "test", "revision": "main"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialized_file(path: Path, *, dataset: str, split: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "source": "bundle_materialized",
        "dataset": dataset,
        "split": split,
        "path": str(path.resolve()),
        "rows": sum(1 for _ in path.open("r", encoding="utf-8")),
        "sha256": sha256(path),
    }


def main() -> int:
    model_dir = ROOT / "models" / "LLaDA-8B-Instruct"
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        max_workers=8,
    )
    manifest = {
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "local_dir": str(model_dir)},
        "datasets": {},
    }
    cache_dir = ROOT / "datasets" / "hf_cache"
    for name, spec in DATASETS.items():
        dataset = load_dataset(
            spec["path"], spec["config"], split=spec["split"],
            revision=spec["revision"], cache_dir=str(cache_dir),
        )
        manifest["datasets"][name] = {
            **spec,
            "rows": len(dataset),
            "fingerprint": dataset._fingerprint,
            "cache_files": [item.get("filename") for item in dataset.cache_files],
        }
    # Optional local assets are recorded only when supplied by the user. The
    # release does not require ARC/PIQA files for its four-task runner.
    optional_assets = {
        "arc_challenge": (ROOT / "official_protocol_repair_campaign" / "data" / "arc_challenge" / "test.jsonl",
                           "ai2_arc/ARC-Challenge", "test"),
        "piqa": (ROOT / "official_protocol_repair_campaign" / "data" / "piqa" / "valid_with_labels.jsonl",
                  "ybisk/piqa", "validation"),
    }
    for name, (path, dataset_name, split) in optional_assets.items():
        if path.exists():
            manifest["datasets"][name] = materialized_file(path, dataset=dataset_name, split=split)
        else:
            manifest["datasets"][name] = {"status": "not_supplied", "dataset": dataset_name, "split": split}
    output = ROOT / "artifacts" / "assets_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
