"""MATH-500 rendering and scoring pinned to this reproduction project."""

from __future__ import annotations

import re
from typing import Any

from math_verify import parse, verify


def doc_to_text(doc: dict[str, Any]) -> str:
    return f"Problem:\n{doc['problem']}\n\nSolution:"


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index, character in enumerate(text[start + len("\\boxed"):], start + len("\\boxed")):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{"):index]
    return None


def _answer(doc: dict[str, Any]) -> str:
    return str(doc.get("answer") or _last_boxed(str(doc.get("solution", ""))) or "")


def _normalize(value: str) -> str:
    value = value.strip()
    value = value.replace("$", "").replace("\\,", "").replace(" ", "")
    value = re.sub(r"^\\boxed\{(.*)\}$", r"\1", value)
    return value


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    prediction = results[0]
    gold = _answer(doc)
    predicted_answer = _last_boxed(prediction) or prediction
    try:
        # A bare tuple such as ``\left(3, \pi/2\right)`` is otherwise parsed as
        # its first number; boxing supplies an unambiguous math extraction anchor.
        verified = int(bool(verify(parse(f"$\\boxed{{{gold}}}$"), parse(prediction))))
    except Exception:
        verified = 0
    return {"exact_match": int(_normalize(gold) == _normalize(predicted_answer)), "math_verify": verified}
