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
| `candidate_loss_gt_tolerance_count` | 18 |
| `ablation_candidate_dominated_scenario_count` | 0 |

## Online Distribution

| Metric | Geomean | Median | Worst | 5% | 95% | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `candidate_vs_legacy_makespan` | 1.54909 | 1.10057 | 1.00352 | 1.01449 | 9.83459 | 1.00352 | 11.8011 | 192 |
| `candidate_vs_legacy_mean_flow` | 3.94398 | 2.4648 | 0.998809 | 1.04001 | 179.178 | 0.998809 | 206.006 | 192 |
| `candidate_mean_backlog_jobs` | 7.40816 | 8.22432 | 1.09946 | 1.69778 | 27.7575 | 1.09946 | 48.112 | 192 |
| `candidate_max_backlog_jobs` | 23.5624 | 25 | 4 | 6 | 67 | 4 | 90 | 192 |

## Scope

Summarizes policy-semantics replay on the same measured service cache. Rows are scenario-level replay summaries, not direct external binary runs.
