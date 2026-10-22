from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from .data import ROOT, digest


def atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def sources():
    paths = list((ROOT / "multimodel_repair").glob("*.py"))
    paths += [ROOT / p for p in ["dacs_method_v5/sampler.py", "dacs_method_v5/policy.py",
              "dacs_method_v5/policies.json", "official_protocol_repair_campaign/official_sampler.py",
              "configs/lm_eval_tasks/math500_utils.py", "vendor/Fast-dLLM/v1/llada/generate.py",
              "vendor/Fast-dLLM/v1/dream/model/generation_utils_block.py",
              "dacs_method_experiment/policy.py", "dacs_method_experiment/sampler.py", "vendor/LLaDA/generate.py",
              "official_protocol_repair_campaign/configs/lm_eval_tasks/mbpp_utils.py",
              "artifacts/dacs_v5_multimodel_adapter_audit.json",
              "configs/lm_eval_tasks/math500_utils.py",
              "configs/lm_eval_tasks/gsm8k_cot.yaml",
              "official_protocol_repair_campaign/configs/lm_eval_tasks/mbpp_utils.py",
              "setcommit/policy.py", "setcommit/selector.py", "setcommit/eb.py"]]
    for model in ("LLaDA-8B-Instruct", "LLaDA2.2-mini", "LLaDA2.1-mini", "Dream-v0-Instruct-7B"):
        paths.extend((ROOT / "models" / model).glob("*.py"))
        paths.extend((ROOT / "models" / model).glob("*.json"))
    paths.extend((ROOT / "vendor/Fast-dLLM/v1/llada/model").glob("*.py"))
    paths.extend((ROOT / "vendor/Fast-dLLM/v1/dream/model").glob("*.py"))
    existing = {p.resolve() for p in paths if p.exists() and p.is_file()}
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(existing)}


def source_hash():
    return hashlib.sha256(json.dumps(sources(), sort_keys=True).encode()).hexdigest()


def environment():
    def command(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=False)
        except OSError as exc:
            return f"unavailable: {exc}"
        return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"
    package_names = ("torch", "transformers", "datasets", "math-verify", "numpy")
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages,
            "gpu": command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"]),
            "cpu": command(["lscpu"]), "memory": command(["free", "-b"]),
            "disk": command(["df", "-B1", str(ROOT)]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "offline": True}
