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
| `candidate_loss_gt_tolerance_count` | 43 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.61023 | 1.23161 | 1.00276 | 1.00703 | 9.83459 | 1.00276 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 4.01799 | 2.49959 | 0.998809 | 1.01215 | 179.178 | 0.998809 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.84293 | 8.46064 | 1.09946 | 1.7673 | 28.2143 | 1.09946 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 24.0034 | 25 | 4 | 6.55 | 67 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
