"""Pinned local Arrow assets and unchanged, dependency-free prompt rendering."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from functools import lru_cache
from numbers import Integral
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"gsm8k_cot": (1319, 256), "math500": (500, 512),
         "mbpp": (500, 256), "humaneval": (164, 512)}
STOPS = {"gsm8k_cot": ("Question:", "<|eot_id|>", "</s>", "<|im_end|>"),
         "math500": ("Problem:",), "mbpp": ("[DONE]",), "humaneval": ()}


def truncate_output(task: str, raw_text: str) -> str:
    for stop in STOPS[task]:
        raw_text = raw_text.split(stop, 1)[0]
    return raw_text


def normalize_eos_token_ids(eos_token_ids) -> list[int]:
    if hasattr(eos_token_ids, "tolist"):
        eos_token_ids = eos_token_ids.tolist()
    if eos_token_ids is None:
        return []
    if isinstance(eos_token_ids, Integral):
        return [int(eos_token_ids)]
    return [int(token) for token in eos_token_ids]


def truncate_ids_at_eos(ids, eos_token_ids):
    """Return the prefix before the first model EOS and its position."""
    eos_token_ids = normalize_eos_token_ids(eos_token_ids)
    if not eos_token_ids:
        return ids, None
    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    eos = set(eos_token_ids)
    positions = [index for index, token in enumerate(values) if token in eos]
    first = min(positions) if positions else None
    return (ids if first is None else ids[:first]), first


def decode_output(tokenizer, ids, task, eos_token_ids=None):
    # Model EOS is an implicit generate-until boundary. Apply it at token level
    # before task strings and before removing special tokens, or post-EOS text
    # can be concatenated into the answer and produce a false positive.
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    prefix, first_eos = truncate_ids_at_eos(
        ids, tokenizer.eos_token_id if eos_token_ids is None else eos_token_ids
    )
    stopped = truncate_output(task, tokenizer.decode(prefix, skip_special_tokens=False))
    visible = tokenizer.decode(tokenizer(stopped, add_special_tokens=False)["input_ids"], skip_special_tokens=True)
    normalized_eos = normalize_eos_token_ids(
        tokenizer.eos_token_id if eos_token_ids is None else eos_token_ids
    )
    return visible, raw, {"first_eos_index": first_eos, "eos_token_ids": normalized_eos,
                          "order": "token_eos_then_task_strings_then_skip_special_tokens"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(None)
def dataset(task: str, split: str = "test"):
    from datasets import Dataset
    manifest = json.loads((ROOT / "artifacts/assets_manifest.json").read_text())
    entry = manifest["datasets"]["gsm8k" if task == "gsm8k_cot" else task]
    path = Path(entry["cache_files"][0])
    if split != "test":
        path = path.with_name(path.name.replace("-test.arrow", f"-{split}.arrow"))
    ds = Dataset.from_file(str(path))
    if split == "test" and len(ds) != TASKS[task][0]:
        raise ValueError(f"unexpected {task} dataset size: {len(ds)}")
    return ds


def asset_record(task: str) -> dict:
    ds = dataset(task)
    path = Path(ds.cache_files[0]["filename"])
    cache_info = json.loads(path.with_name("dataset_info.json").read_text())
    download_urls = list(cache_info.get("download_checksums", {}))
    revisions = {m.group(1) for url in download_urls if (m := re.search(r"@([0-9a-f]{40})/", url))}
    if len(revisions) != 1:
        raise ValueError(f"cannot establish unique dataset revision from local metadata: {task}")
    order = hashlib.sha256()
    for row in ds:
        order.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        order.update(b"\n")
    return {"path": str(path), "split": "test", "rows": len(ds),
            "fingerprint": ds._fingerprint, "arrow_sha256": digest(path),
            "sample_order_sha256": order.hexdigest(), "shuffle": False,
            "upstream_revision": next(iter(revisions)), "cached_source_urls": download_urls,
            "revision_note": "Recovered from revision-qualified hf:// download URLs in dataset_info.json; content also pinned by SHA256."}


@lru_cache(None)
def mbpp_demonstrations() -> list[dict]:
    # Extract the literal only: importing lm_eval.tasks.mbpp also imports model
    # classes incompatible with Transformers 5 and can initialize network metrics.
    lm_eval_spec = importlib.util.find_spec("lm_eval")
    if lm_eval_spec is None or not lm_eval_spec.submodule_search_locations:
        raise RuntimeError("MBPP few-shot rendering requires lm-eval==0.4.8; install requirements-full.txt")
    path = Path(next(iter(lm_eval_spec.submodule_search_locations))) / "tasks/mbpp/utils.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "list_fewshot_samples")
    value = next(n.value for n in fn.body if isinstance(n, ast.Return))
    samples = ast.literal_eval(value)
    local = ROOT / "official_protocol_repair_campaign/configs/lm_eval_tasks/mbpp_utils.py"
    tree = ast.parse(local.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "append")
    samples.append(ast.literal_eval(call.args[0]))
    if [x["task_id"] for x in samples] != [2, 3, 4, 5]:
        raise ValueError("MBPP four-shot examples changed")
    return samples


@lru_cache(None)
def fewshot(task: str) -> str:
    if task == "mbpp":
        return "\n\n".join(mbpp_question(d) + d["code"].rstrip() + "\n[DONE]" for d in mbpp_demonstrations())
    return ""


@lru_cache(None)
def gsm8k_paper_contexts() -> tuple[str, ...]:
    """Exact lm-eval 0.4.8 GSM8K five-shot request order (seed 1234)."""
    import random

    training = list(dataset("gsm8k_cot", "train"))
    test = dataset("gsm8k_cot")
    rng = random.Random(1234)
    contexts = []
    for doc in test:
        examples = rng.sample(training, 5)
        pieces = [
            f"Question: {example['question']}\nAnswer: {example['answer']}"
            for example in examples
        ]
        pieces.append(f"Question: {doc['question']}\nAnswer:")
        contexts.append("\n\n".join(pieces))
    return tuple(contexts)


def mbpp_question(d: dict) -> str:
    return ("You are an expert Python programmer, and here is your task: " + d["text"]
            + " Your code should pass these tests:\n\n" + "\n".join(d["test_list"][:3]) + "\n[BEGIN]\n")


def prompt_for(task: str, d: dict, *, index: int | None = None) -> str:
    if task == "gsm8k_cot":
        if index is None:
            raise ValueError("GSM8K paper contexts require the test-set index")
        expected = dataset(task)[index]
        if expected["question"] != d["question"] or expected["answer"] != d["answer"]:
            raise ValueError("GSM8K document/index mismatch")
        return gsm8k_paper_contexts()[index]
    if task == "math500":
        return f"Problem:\n{d['problem']}\n\nSolution:"
    if task == "mbpp":
        return fewshot(task) + "\n\n" + mbpp_question(d)
    if task == "humaneval":
        return str(d["prompt"])
    raise ValueError(task)
