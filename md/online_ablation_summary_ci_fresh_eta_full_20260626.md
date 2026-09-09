# Online Replay and Ablation Summary Intervals

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `ONLINE_ABLATION_DISTRIBUTIONAL_SUMMARY_PASS` |
| `online_scenario_count` | 192 |
| `ablation_scenario_count` | 24 |
| `online_candidate_dominated_scenario_count` | 5 |
| `candidate_loss_gt_tolerance_count` | 112 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.6097 | 1.23161 | 1.00276 | 1.00654 | 9.83459 | 1.00276 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 3.99567 | 2.49959 | 0.989192 | 1.00844 | 179.178 | 0.989192 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.88417 | 8.46064 | 1.09946 | 1.7673 | 28.2143 | 1.09946 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 24.1022 | 25 | 4 | 7 | 67 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
