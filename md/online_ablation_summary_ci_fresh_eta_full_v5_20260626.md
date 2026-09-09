# Online Replay and Ablation Summary Intervals

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `ONLINE_ABLATION_DISTRIBUTIONAL_SUMMARY_PASS` |
| `online_scenario_count` | 192 |
| `ablation_scenario_count` | 24 |
| `online_candidate_dominated_scenario_count` | 2 |
| `candidate_loss_gt_tolerance_count` | 119 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.61834 | 1.24267 | 1.00276 | 1.00749 | 9.83459 | 1.00276 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 4.16428 | 2.49182 | 1.00048 | 1.02678 | 179.178 | 1.00048 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.8121 | 8.32989 | 1.09946 | 1.76326 | 27.6566 | 1.09946 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 23.7285 | 24 | 4 | 7 | 65 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
