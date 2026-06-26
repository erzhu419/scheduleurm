# Online Replay and Ablation Summary Intervals

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `ONLINE_ABLATION_DISTRIBUTIONAL_SUMMARY_PASS` |
| `online_scenario_count` | 192 |
| `ablation_scenario_count` | 24 |
| `online_candidate_dominated_scenario_count` | 0 |
| `candidate_loss_gt_tolerance_count` | 68 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.61844 | 1.24849 | 1.00276 | 1.00749 | 9.83459 | 1.00276 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 4.1801 | 2.49182 | 1.00048 | 1.02678 | 179.178 | 1.00048 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.78299 | 8.31831 | 1.09946 | 1.76326 | 27.6566 | 1.09946 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 23.6543 | 24 | 4 | 7 | 65 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
