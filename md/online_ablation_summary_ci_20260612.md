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
| `candidate_loss_gt_tolerance_count` | 106 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.66963 | 1.32933 | 1.0034 | 1.01962 | 9.83459 | 1.0034 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 4.96904 | 3.62865 | 0.989192 | 1.0235 | 179.178 | 0.989192 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.43952 | 8.29288 | 0.941319 | 1.65881 | 27.7575 | 0.941319 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 23.6069 | 25 | 4 | 6 | 67 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
