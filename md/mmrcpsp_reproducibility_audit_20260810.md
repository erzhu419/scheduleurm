# MMRCPSP Reproducibility Audit

- Status: `MMRCPSP_SEMANTIC_REPRODUCIBILITY_PASS`
- Semantic equality is evaluated after decompression and after removing only container-dependent self-hash fields.
- Raw gzip equality is reported separately and is not used to excuse any result drift.

| Experiment | Full JSON after decompression | Compact semantics | Markdown | Raw gzip | Gzip FNAME |
|---|---:|---:|---:|---:|---|
| mmrcpsp_renewal_stream_v1 | True | True | True | False | `mmrcpsp_renewal_stream_v1_20260810.json` vs `mmrcpsp_renewal_stream_v1_repro_20260810.json` |
| mmrcpsp_class_balance_holdout_v1 | True | True | True | False | `mmrcpsp_class_balance_holdout_v1_20260810.json` vs `mmrcpsp_class_balance_holdout_v1_repro_20260810.json` |

## Interpretation

- Supports: exact semantic reproduction of both deterministic MMRCPSP experiment matrices under the frozen code and input ledgers.
- Does not support: path-independent bitwise identity of gzip containers whose headers retain different output basenames.
