# Reviewer reproduction checklist

## 1. Selector and composition smoke

```bash
python3 -m pip install -r requirements.txt
PYTHON_BIN=python3 bash scripts/verify_release.sh
```

The smoke is deterministic and model-free. It checks the controlled quality
case, Native/Fast fixed and adaptive host contracts, EB quota construction,
EB+SETCOMMIT composition, equal forward counts, and all net-progress audits.

## 2. Full environment

For the paper's CUDA 12.1 environment (including the explicit `math-verify`
dependency used by MATH-500 scoring):

```bash
python3 -m pip install -r requirements-cu121.txt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.cache/huggingface
python3 prepare_assets.py
```

The checkpoint is pinned to `GSAI-ML/LLaDA-8B-Instruct` revision
`08b83a6feb34df1a6011b80c3c00c7563e963b07`. The generated
`artifacts/assets_manifest.json` records dataset revisions, splits, row counts,
fingerprints and cache files.

## 3. Paired LLaDA-8B cases

```bash
PYTHON_BIN=python3 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_llada_pair.sh
PYTHON_BIN=python3 KIND=fast_fixed STEPS=64 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_llada_pair.sh
PYTHON_BIN=python3 KIND=fast_threshold STEPS=64 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_llada_pair.sh
```

The default limit is two examples for a quick reviewer run. Set `LIMIT` for a
larger paired sample. The runner uses batch size one, BF16, one warm-up and one
timing repeat. Model loading is excluded from latency. Results include paired
outputs, score, NFE, timing and trace/audit fields.

## 4. What must match

Do not change the model revision, tokenizer/chat template, mask/EOS IDs,
prompt construction, seed, scorer, host transfer schedule, or policy JSON when
comparing baseline and SETCOMMIT. Native/Fast uses `NATIVE_FAST`; EB uses
`EB_QUALITY_GUARD` and logs actual NFE separately from effective stop NFE.
