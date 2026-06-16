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
| `node007_30gb_add_small_stable` | 0.0288151 | 0.0540705 | 46.71% | 0.0512251 | 0.0575856 | 11.05% | 2373.63 | true |
| `cpu_half_node001_add_single_v2` | 0.752276 | 0.987992 | 23.86% | 0.96672 | 1.28101 | 24.53% | 17.2669 | true |
| `cpu_full_node001_add_single_v2` | 1.05372 | 1.46448 | 28.05% | 1.52121 | 1.7575 | 13.44% | 10.2437 | true |

## Scope

The holdout compares immediate co-location with defer-until-resident for the measured corner-case rows.  The resident-alone challenge rate uses the fastest observed resident-alone window, which favors the defer baseline.  This is still a scoped measured-service holdout, not a direct full-stack Gavel/Pollux/Sia/IADeep/Salus execution result.
