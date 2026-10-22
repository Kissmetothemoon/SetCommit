"""Offline deterministic scoring; never inspect tests to choose an extraction."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

GSM8K_FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
GSM8K_IGNORES = (",", r"\$", r"(?s).*#### ", r"\.$")


@lru_cache(None)
def code_executor():
    """Return a self-contained deterministic pass/fail executor.

    The release must not depend on a machine-specific Hugging Face evaluate
    cache path. A subprocess gives each completion an isolated interpreter and
    enforces the same wall-clock timeout used by the campaign scorer.
    """
    def check_correctness(program: str, timeout: float, task_id: str, completion_id: int):
        with tempfile.TemporaryDirectory(prefix="setcommit-codeeval-") as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(program, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)], capture_output=True,
                    text=True, timeout=float(timeout), check=False,
                )
                return {"passed": proc.returncode == 0,
                        "result": "passed" if proc.returncode == 0 else "failed",
                        "task_id": str(task_id), "completion_id": int(completion_id),
                        "stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]}
            except subprocess.TimeoutExpired as exc:
                return {"passed": False, "result": "timed out",
                        "task_id": str(task_id), "completion_id": int(completion_id),
                        "stderr": str(exc), "stdout": ""}
    return check_correctness


def extract_code(text: str) -> str:
    text = text.split("[DONE]", 1)[0]
    if "[BEGIN]" in text:
        text = text.split("[BEGIN]", 1)[1]
    match = re.search(r"```(?:python|py)?[^\S\n]*\n(.*?)```", text, re.S | re.I)
    # Preserve leading indentation: a HumanEval completion may be a function body.
    return (match.group(1) if match else text).strip("\r\n")


def code_program(task: str, doc: dict, output: str) -> tuple[str, str]:
    code = extract_code(output)
    mode = "complete_program"
    if task == "humaneval":
        try:
            tree = ast.parse(code)
            complete = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.name == doc["entry_point"] for n in tree.body)
        except SyntaxError:
            complete = False
        if complete:
            # Prompt imports/helper functions remain available; generated entry
            # point replaces only the stub. No reference solution is included.
            program = doc["prompt"] + "\n" + code
        else:
            mode = "prompt_completion"
            # lm-eval's completion stop rules apply to body continuations.
            # Complete instruct functions use the deterministic module branch.
            for stop in ("\nclass", "\ndef", "\n#", "\nif", "\nprint"):
                code = code.split(stop, 1)[0]
            program = doc["prompt"] + code
        return program + "\n" + doc["test"] + f"\ncheck({doc['entry_point']})\n", mode
    if task == "mbpp":
        setup = doc.get("test_setup_code") or ""
        return setup + "\n" + code + "\n" + "\n".join(doc["test_list"]), mode
    raise ValueError(task)


def last_number(text: str) -> str:
    hits = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return hits[-1].replace(",", "") if hits else ""


def gsm8k_flexible_exact_match(doc: dict, output: str) -> tuple[float, dict]:
    """Mirror configs/lm_eval_tasks/gsm8k_cot.yaml flexible-extract."""
    matches = GSM8K_FLEXIBLE_PATTERN.findall(output)
    prediction = next((value for value in matches[-1] if value), "[invalid]").strip() if matches else "[invalid]"
    gold = str(doc["answer"]).split("####")[-1].strip()
    normalized_prediction, normalized_gold = prediction, gold
    for pattern in GSM8K_IGNORES:
        normalized_prediction = re.sub(pattern, "", normalized_prediction)
        normalized_gold = re.sub(pattern, "", normalized_gold)
    matched = normalized_prediction.lower() == normalized_gold.lower()
    return float(matched), {"gold": gold, "prediction": prediction,
            "normalized_gold": normalized_gold, "normalized_prediction": normalized_prediction,
            "format_valid": bool(matches), "metric": "exact_match,flexible-extract"}


def score(task: str, doc: dict, output: str) -> tuple[float, dict]:
    if task == "gsm8k_cot":
        return gsm8k_flexible_exact_match(doc, output)
    if task == "math500":
        from configs.lm_eval_tasks.math500_utils import process_results, _last_boxed
        result = process_results(doc, [output])
        return float(result["math_verify"]), {**result, "gold": doc["answer"],
                "prediction": _last_boxed(output), "format_valid": _last_boxed(output) is not None,
                "metric": "math_verify"}
    if task in {"mbpp", "humaneval"}:
        program, mode = code_program(task, doc, output)
        try:
            ast.parse(program)
            valid = True
        except SyntaxError:
            valid = False
        result = code_executor()(program, 8.0, str(doc.get("task_id", "")), 0)
        return float(result["passed"]), {**result, "format_valid": valid, "extraction": mode,
                "metric": "pass@1", "executor": "python_subprocess_isolated"}
    raise ValueError(task)
