# SETCOMMIT: Coordinated Token Commitments

This document is the implementation-facing description used by the ICLR 2027
paper release. The public method name is **SETCOMMIT**. The legacy internal
campaign identifier `dacs_v5_exchange_020` maps exactly to the
`NATIVE_FAST` policy below and is retained only for result provenance.

## Interface and scope

At each host-decoder forward, the host provides:

* logits `z[i, v]` for the current sequence;
* the current masked/visible state `x`;
* the host top-1 proposal `y[i]`;
* the host transfer quota `q_t` and editable positions.

The public NumPy API receives an all-mask mutable generation block by default.
If visible tokens are present, callers must pass `editable_positions`; the
selector never infers that prompt positions are safe to edit.

SETCOMMIT returns a set of `q_t` net commitments and an optional equal-sized
remask set. It does not change model weights, logits, the diffusion schedule,
the host stopping rule, or the number of model forwards. In a Fast-dLLM host,
the host continues to own dual-cache construction and refresh; SETCOMMIT is
called after the host forward and before the replacement-position mask is
applied.

## Selection score

For each masked position, let `p_i` be the softmax distribution, `c_i = p_i(y_i)`
the top-1 posterior, and `m_i = z_i,(1) - z_i,(2)` the logit margin. Margins
are min-max normalized over masked candidates. The Native/Fast policy ranks the
shortlist by normalized margin; the EB policy ranks by posterior.

Normalized entropy is

```text
u_i = - sum_v p_i(v) log p_i(v) / log |V|.
```

The implementation risk kernel is

```text
a_ij = d_t exp(-|i-j| / tau_t) u_i u_j, i != j;   a_ii = 0,
d_t = |M_t| / sequence_length.
```

The schedules are `lambda_t = 0.03 - 0.02 s_t` and `tau_t = 16 - 12 s_t`,
where `s_t=(t+1)/T` is supplied by the host. A greedy candidate receives

```text
g_t(i | S) = cbar_i - min(lambda_t * sum_{j in S} a_ij, 0.10).
```

The candidate pool is the first `ceil(4 q_t)` margin-ranked masked positions,
augmented with the posterior top-`q_t` baseline. Greedy selection reserves
enough baseline positions to retain at least `ceil(0.75 q_t)` overlap. The
selection is cardinality checked; an invalid host quota is a failed run, never
a silent fallback.

## Bounded exchange

After the base set is selected, a visible position may be withdrawn only when
its old token differs from the current top-1, is outside the current top-2,
and the new posterior exceeds the old posterior by the policy threshold.
Withdrawals are ordered by increasing old posterior. Fresh positions are
selected from the remaining masked pool by the same selector. Candidate
exchange sizes are tried from `floor(r q_t)` down to one (`r=0.20` for
Native/Fast; `r=0.10` for EB), and are accepted only when:

1. every paired posterior gain is above the configured threshold;
2. the set-risk audit does not increase (tolerance `1e-7`);
3. the active host block and structural protection rules are respected.

If `R_t` is remasked and `A_t` is newly committed, the host-visible invariant
is always checked:

```text
|A_t| - |R_t| = q_t.
```

The implementation applies remasks first and commits the host top-1 proposals
second. This makes cache replacement explicit for Fast-dLLM while retaining
the host's cache ownership.

## EB-Sampler composition

EB first orders masked positions by its error proxy (LLaDA: `1-confidence`;
Dream: entropy). For threshold `gamma`, it selects the longest prefix `k`
satisfying

```text
sum_{j<=k} h_j - max_{j<=k} h_j <= gamma.
```

That prefix is the EB baseline and determines `q_t`. SETCOMMIT then receives
that exact quota and baseline using `EB_QUALITY_GUARD` (`2q` shortlist, 90%
baseline retention, posterior reliability, ratio 0.10, gain 0.005, window
[0.10,0.60], structural protection). The selected set need not itself satisfy
the EB entropy bound; both the original and selected-set bounds are logged.
Actual NFE counts every forward to completion. Effective NFE is an additional
stop-position diagnostic and must never replace actual NFE or latency.

## Pseudocode

```text
for each host forward t:
    logits, state, quota, editable = host.forward()
    baseline = host.baseline(logits, state, quota)  # top-k or EB prefix
    S, trace = SETCOMMIT(logits, state, baseline, quota, policy)
    assert len(S) - len(trace.remasked) == quota
    state = host.apply_remask_and_commit(S, trace.remasked)
    host.refresh_cache(trace.remasked)  # Fast-dLLM only
```

The package's NumPy implementation is deterministic under tied logits (stable
position-index tie breaking), emits a per-step trace, and performs no extra
model forward.
