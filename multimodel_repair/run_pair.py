"""Measure paired methods on the same loaded model/GPU; checkpoint every prompt."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch

from .common import atomic, sources, environment, source_hash
from .data import dataset, prompt_for, TASKS, asset_record, decode_output, STOPS
from .generation import Generator
from .scoring import score


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def statistics(values):
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def reconcile_trace_checkpoint(checkpoint: Path, traces: Path, samples: list[dict]) -> None:
    """Trim an uncommitted trace row left by an interrupted append."""
    expected = list(range(len(samples)))
    if [row.get("index") for row in samples] != expected:
        raise RuntimeError("partial sample checkpoint is not contiguous")
    if not traces.exists():
        if samples:
            raise RuntimeError("partial sample checkpoint has no matching trace")
        return
    with gzip.open(traces, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if [row.get("index") for row in rows[:len(samples)]] != expected:
        raise RuntimeError("partial trace does not match the sample checkpoint")
    if len(rows) == len(samples):
        return
    if len(rows) < len(samples):
        raise RuntimeError("partial trace is shorter than the sample checkpoint")
    tmp = traces.with_name(traces.name + f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as stream:
        for row in rows[:len(samples)]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, traces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--kind", choices=["fixed", "fast_fixed", "fast_threshold", "reference"], default="fixed")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timing-repeats", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.timing_repeats < 1:
        ap.error("--timing-repeats must be >= 1")
    torch.set_num_threads(4)
    methods = {"fixed": ["topk_uniform", "dacs_v5_exchange_020"],
               "fast_fixed": ["fast_dual_fixed", "fast_dual_fixed_dacs"],
               "fast_threshold": ["fast_dual_threshold", "fast_dual_threshold_dacs"],
               "reference": ["native_reference"]}[args.kind]
    started = time.time()
    record = {"schema": 3, "status": "running", **vars(args), "output": str(args.output),
              "sources": sources(), "source_hash": source_hash(), "samples": [],
              "warmup": args.warmup, "timing_repeats": args.timing_repeats, "length": TASKS[args.task][1],
              "stop_strings": STOPS[args.task],
              "decode_protocol": "token EOS, then task stop strings, then remove special tokens",
              "timing_scope": "generation only; excludes load, tokenization, decode, scoring and serialization"}
    checkpoint = args.output.with_suffix(".samples.jsonl")
    traces = args.output.with_suffix(".trace.jsonl.gz")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and json.loads(args.output.read_text()).get("status") == "ok":
            raise RuntimeError("refusing to overwrite completed results; use a new campaign directory")
        identity_path = args.output.with_suffix(".identity.json")
        if checkpoint.exists():
            previous = json.loads(identity_path.read_text())
            if previous["source_hash"] != record["source_hash"]:
                raise RuntimeError("partial result code changed; use a new campaign directory")
            for field in ("model", "task", "steps", "kind", "limit", "seed", "output"):
                if previous.get(field) != record[field]:
                    raise RuntimeError(f"partial result {field} does not match this job")
            record["samples"] = [json.loads(line) for line in checkpoint.read_text().splitlines() if line]
        reconcile_trace_checkpoint(checkpoint, traces, record["samples"])
        atomic(identity_path, {k: v for k, v in record.items() if k != "samples"})
        record["environment"] = environment()
        record["assets"] = asset_record(args.task)
        atomic(args.output, record)
        seed(args.seed)
        gen = Generator(args.model, fast=args.kind.startswith("fast_"))
        record["identity"] = gen.identity()
        docs = dataset(args.task)
        count = min(args.limit, len(docs))
        length = record["length"]
        warm_ids = gen.encode(prompt_for(args.task, docs[0], index=0))
        for method in methods:
            for _ in range(args.warmup):
                seed(args.seed)
                gen(warm_ids, method=method, steps=args.steps, length=length)
        torch.cuda.synchronize()
        for idx in range(len(record["samples"]), count):
            doc = docs[idx]
            prompt = prompt_for(args.task, doc, index=idx)
            prompt_ids = gen.encode(prompt)
            row = {"index": idx, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                   "input_ids_hash": hashlib.sha256(prompt_ids.cpu().numpy().tobytes()).hexdigest(),
                   "methods": {}}
            all_traces = {}
            for method in methods:
                times, hashes, nfes, memories = [], [], [], []
                first = None
                for repetition in range(args.timing_repeats):
                    seed(args.seed)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    result = gen(prompt_ids, method=method, steps=args.steps, length=length)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - begin)
                    memories.append(torch.cuda.max_memory_allocated() / 1024**2)
                    hashes.append(hashlib.sha256(result.ids.cpu().numpy().tobytes()).hexdigest())
                    nfes.append(result.audit["actual_nfe"])
                    if first is None:
                        first = result
                if len(set(hashes)) != 1 or len(set(nfes)) != 1:
                    raise RuntimeError(f"non-deterministic repeated output: {method} sample {idx}")
                output, raw_output, decode = decode_output(gen.tokenizer, first.ids, args.task, gen.eos)
                value, detail = score(args.task, doc, output)
                row["methods"][method] = {"score": value, "detail": detail, "output": output,
                    "raw_output": raw_output, "decode": decode,
                    "generated_token_ids": first.ids.cpu().tolist(),
                    "visible_output_tokens": len(gen.tokenizer(output, add_special_tokens=False)["input_ids"]),
                    "timing_seconds": times, "latency": statistics(times), "actual_nfe": nfes[0],
                    "token_ids_sha256": hashes[0], "all_repeat_hashes": hashes,
                    "generated_token_slots": len(first.ids),
                    "peak_memory_mib": max(memories), "audit": first.audit}
                all_traces[method] = first.trace
            if args.kind in {"fixed", "fast_fixed"}:
                schedules = [[r["net_commits"] for r in all_traces[m]] for m in methods]
                if schedules[0] != schedules[1]:
                    raise RuntimeError("paired methods used different transfer schedules")
                row["paired_transfer_schedule_equal"] = True
            with gzip.open(traces, "at", encoding="utf-8") as stream:
                stream.write(json.dumps({"index": idx, "trace": all_traces}, ensure_ascii=False) + "\n")
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            record["samples"].append(row)
            atomic(args.output.with_suffix(".progress.json"), {"samples": len(record["samples"]),
                        "total": count, "updated": time.time(), "elapsed_seconds": time.time() - started})
            print(json.dumps({"model": args.model, "task": args.task, "kind": args.kind,
                              "steps": args.steps, "completed": idx + 1, "total": count}), flush=True)
        summary = {}
        for method in methods:
            rows = [r["methods"][method] for r in record["samples"]]
            times = [t for r in rows for t in r["timing_seconds"]]
            summary[method] = {"accuracy": float(np.mean([r["score"] for r in rows])),
                "latency": statistics(times), "generation_seconds_mean_repeat": sum(np.mean(r["timing_seconds"]) for r in rows),
                "format_valid_rate": float(np.mean([r["detail"]["format_valid"] for r in rows])),
                "peak_memory_mib": max(r["peak_memory_mib"] for r in rows),
                "actual_nfe_mean": float(np.mean([r["actual_nfe"] for r in rows])),
                "tokens_sec": sum(r["visible_output_tokens"] for r in rows) / sum(np.mean(r["timing_seconds"]) for r in rows),
                "generated_slots_sec": sum(r["generated_token_slots"] for r in rows) / sum(np.mean(r["timing_seconds"]) for r in rows),
                "tokens_per_actual_forward": sum(r["visible_output_tokens"] for r in rows) / sum(r["actual_nfe"] for r in rows),
                "exchange_count": sum(r["audit"]["exchange_count"] for r in rows),
                "net_commits": sum(r["audit"]["net_commits"] for r in rows)}
        gate = {"passed": True, "max_slowdown": 1.5}
        if len(methods) == 2:
            ratios = {p: summary[methods[1]]["latency"][p] / summary[methods[0]]["latency"][p] for p in ["mean", "p50", "p95"]}
            gate.update(slowdown=ratios, passed=all(v <= 1.5 for v in ratios.values()))
        record.update(status="ok", summary=summary, speed_gate=gate, trace_file=str(traces),
                      elapsed_seconds=time.time() - started, completed=time.time())
        atomic(args.output, record)
        print(json.dumps({"status": "ok", "summary": summary, "speed_gate": gate}), flush=True)
        return 0
    except Exception as exc:
        record.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        # Do not overwrite a valid historical result even on misuse.
        destination = args.output
        if destination.exists() and json.loads(destination.read_text()).get("status") == "ok":
            destination = args.output.with_suffix(".rejected.json")
        atomic(destination, record)
        print(record["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
