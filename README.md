# SetCommit: Coordinated Token Commitments

Official code release for the paper **SetCommit: Coordinated Token Commitments
for Few-Step Diffusion Language Model Decoding**.

Yuannuo Feng, Yubing Ye, Wenshuai Yao, Yizhe Chen, Ngai Wong, Wenyong Zhou,
Wang Kang

This is the paper-aligned reproduction package for
**SetCommit**, a training-free and modular commitment-selection layer for
masked diffusion language models. It coordinates *which* tokens are committed
at an upstream decoder's existing budget. It adds no model forward, requires no
training, and composes with Native sampling, Fast-dLLM fixed/adaptive modes,
and the paper's entropy-bounded (EB) sampler.

The internal experiment ID `dacs_v5_exchange_020` is preserved in manifests and
configuration files for exact result provenance; it is an alias for the public
`NATIVE_FAST` SETCOMMIT policy, not a second method.

## Quick start (no model download)

```bash
cd SetCommit
python3 -m pip install -r requirements.txt
bash scripts/run_smoke.sh
```

The deterministic CPU smoke checks a controlled low-NFE quality-selection case,
Native/Fast-dLLM fixed and adaptive host contracts, the EB quota rule and
EB+SETCOMMIT composition, equal forward counts, cache ownership, and the
net-progress invariant. It uses instrumented fake host counters (not filled-in
telemetry), verifies a visible-token remask, and checks the EB
`sum(error_proxy)-max(error_proxy)` bound and 90% retention. It writes
`examples/smoke_result.json` and exits
non-zero on any audit failure.

## Public API

```python
import numpy as np
from setcommit import NATIVE_FAST, select_positions, apply_selection
from setcommit import entropy_bound_baseline, EB_QUALITY_GUARD

selected, trace = select_positions(
    logits, state, predicted, host_baseline,
    mask_id=mask_id, step=t, steps=T, config=NATIVE_FAST,
    editable_positions=np.arange(generation_block_length),
)
state = apply_selection(state, predicted, selected, trace, mask_id=mask_id)
```

`trace` records baseline/base/selected/remasked positions, overlap, risk,
exchange cap, posterior gains, and the exact cardinality audit:
`new_commits - remasked = upstream_quota`.
The API operates on an all-mask generation block by default. Once `state`
contains visible tokens it requires `editable_positions`; prompt positions
are then ineligible for remasking.

## Paper policies

| Host | Public config | Internal ID | Key settings |
|---|---|---|---|
| Native / Fast-dLLM | `NATIVE_FAST` | `dacs_v5_exchange_020` | margin, 4q shortlist, 75% overlap, exchange 0.20, gain 0.002, window [0,.8] |
| EB-Sampler | `EB_QUALITY_GUARD` | `dacs_v5_quality_guard` | posterior, 2q shortlist, 90% overlap, exchange 0.10, gain 0.005, window [.1,.6], structural protection |

Exact JSON policies are in `configs/`; formulas and pseudocode are in
`METHOD.md`.

## LLaDA-8B paired reproduction

The package includes the pinned LLaDA/Fast-dLLM integration and a runner for
the paper's primary paired case. Model weights and datasets are fetched
separately; no weights are redistributed.

For the full CUDA 12.1 runner, install `requirements-cu121.txt` instead of the
CPU-only smoke requirements.

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$PWD/.cache/huggingface
python3 prepare_assets.py
CUDA_VISIBLE_DEVICES=0 bash scripts/run_llada_pair.sh                 # native
KIND=fast_fixed STEPS=64 CUDA_VISIBLE_DEVICES=0 bash scripts/run_llada_pair.sh
```

Use batch size 1, BF16, the pinned revision in `RELEASE_MANIFEST.json`, and
the same prompt/scoring chain for each paired method. Fast-dLLM continues to
own dual-cache creation, replacement-position refresh, threshold stopping and
block boundaries; SETCOMMIT only edits the host's commit/remask set.

## Paper-aligned claims and result anchors

SETCOMMIT is evaluated at matched upstream budgets: it preserves the host
quota and does not add a forward. The included reviewer anchor is:

| Model / task | Host protocol | baseline | SETCOMMIT | gain |
|---|---|---:|---:|---:|
| LLaDA-8B GSM8K | native, 64 → 48 NFE | 73.1% | 74.2% | +1.1pp |

These are the matched-quality operating points from paper Table 1 (native
host, 3-point margin): the baseline meets the quality target at 64
evaluations, SETCOMMIT at 48. The executable package anchor in
`examples/observed_quality_anchors.json` is a separate fixed 64-forward
campaign measurement (65.5% → 68.8%), not the same run; the protocol
crosswalk is in `RESULT_PROVENANCE.md`.

## Layout

* `setcommit/` — public selector, policies, EB adapter and host contract.
* `METHOD.md` — paper Appendix C/D implementation specification.
* `RESULT_PROVENANCE.md` — package-anchor versus paper-Table-1 crosswalk.
* `COMPATIBILITY.md` — mapping for retained historical runner imports.
* `configs/` — Table 3 policies and legacy-ID mapping.
* `examples/run_smoke.py` — deterministic reviewer case.
* `third_party/` and `vendor/` — Fast-dLLM snapshot and notices.
* `scripts/verify_release.sh` — smoke plus manifest validation.

License: MIT for the SetCommit code. Third-party components retain their
upstream licenses and notices.

## Citation

```bibtex
@article{feng2026setcommit,
  title   = {SetCommit: Coordinated Token Commitments for Few-Step Diffusion Language Model Decoding},
  author  = {Feng, Yuannuo and Ye, Yubing and Yao, Wenshuai and Chen, Yizhe and Wong, Ngai and Zhou, Wenyong and Kang, Wang},
  journal = {arXiv preprint},
  year    = {2026}
}
```
