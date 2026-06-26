# Gavel Resident-Delay/JCT Holdout Gate

This is a scoped Gavel-style finish-time holdout over measured Scheduleurm corner-case rows. It is not a direct full-stack SOTA execution claim.

| Quantity | Value |
|---|---:|
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `usable_row_count` | 3 |
| `admit_now_jct_better_count` | 3 |
| `admit_now_makespan_better_count` | 3 |

## Rows

| Scenario | Co-run mean JCT | Defer mean JCT | JCT gain | Co-run makespan | Defer makespan | Makespan gain | Resident challenge | Ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `node007_30gb_add_small_stable` | 0.2174 | 0.321904 | 32.46% | 0.225234 | 0.423566 | 46.82% | 998.899 | true |
| `cpu_half_node001_add_single_v2` | 0.0437045 | 0.0526724 | 17.03% | 0.0576227 | 0.0812366 | 29.07% | 41.4797 | true |
| `cpu_full_node001_add_single_v2` | 0.0533485 | 0.0540942 | 1.38% | 0.0575261 | 0.0826584 | 30.40% | 39.1696 | true |

## Scope

The holdout compares immediate co-location with defer-until-resident for the measured corner-case rows.  The resident-alone challenge rate uses the fastest observed resident-alone window, which favors the defer baseline.  This is still a scoped measured-service holdout, not a direct full-stack Gavel/Pollux/Sia/IADeep/Salus execution result.
