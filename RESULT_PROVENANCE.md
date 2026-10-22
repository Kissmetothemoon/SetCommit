# Result provenance and paper crosswalk

The JSON file `examples/observed_quality_anchors.json` contains one paired
64-forward LLaDA-8B/GSM8K measurement from the internal
`dacs_v5_multimodel` campaign. It is the executable anchor for the reviewer
case, but it is **not** the
operating points in paper Table 1.

The difference is intentional and protocol-level:

| Artifact | Protocol | LLaDA-8B GSM8K baseline → SETCOMMIT |
|---|---|---:|
| `observed_quality_anchors.json` | fixed 64-forward campaign; source files and hashes are recorded per row | 65.5% → 68.8% |
| Paper Table 1 | matched-quality operating point, selected after target sweep; Native 64 → 48 NFE | 73.1% → 74.2% |

The paper table reports the lowest-latency configuration reaching each
model–task quality target (3-point margin), whereas the package anchor fixes a
64-forward diagnostic point. It must not be compared as if it were
the same run. The anchor JSON records model, task, host, step count, scorer
protocol, policy, source hash, NFE, latency and format-valid rate. Full paper
Table 1 values are retained in `iclr2027_conference.pdf` and the manuscript's
run manifests; this release does not silently relabel preliminary anchors as
paper-table results.

The public smoke is synthetic and only verifies selector behavior and host
contracts. It is not a quality estimate for any benchmark.
