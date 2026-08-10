# MMRCPSP class-balance arrival-tape holdout v1

- Status: `MMRCPSP_CLASS_BALANCE_HOLDOUT_PROTOCOL_PASS`
- Runs: `175/175` protocol PASS
- Ours exact registered-family oracle runs: `35`
- Ours Pareto-nondominated scenario/seeds: `35/35`
- Ours strictly improves every baseline: `0/35`
- Baseline dominates ours: `0/35`
- Frozen costs: total average queue, worst-class average queue, and worst-class final backlog.
- The v1 total-only 0/21 result is retained; this is a new-tape metric holdout over the same known action library.
- Performance does not determine protocol validity or imply positive recurrence.

| Scenario | Seed | Total average queue | Worst-class average queue | Worst-class final backlog | Disposition |
|---|---:|---:|---:|---:|---|
| bursty_nominal | 101 | 5.075625 | 0.39375 | 0 | ours_nondominated_tradeoff_or_tie |
| bursty_nominal | 211 | 7.5375 | 0.790625 | 1 | ours_nondominated_tradeoff_or_tie |
| bursty_nominal | 307 | 5.662083333333 | 0.531666666667 | 1 | ours_nondominated_tradeoff_or_tie |
| bursty_nominal | 401 | 6.681458333333 | 0.694375 | 0 | ours_nondominated_tradeoff_or_tie |
| bursty_nominal | 503 | 4.413958333333 | 0.391041666667 | 1 | ours_nondominated_tradeoff_or_tie |
| load_035 | 101 | 3.445 | 0.324583333333 | 1 | ours_nondominated_tradeoff_or_tie |
| load_035 | 211 | 2.855 | 0.263958333333 | 0 | ours_nondominated_tradeoff_or_tie |
| load_035 | 307 | 3.896458333333 | 0.4025 | 0 | ours_nondominated_tradeoff_or_tie |
| load_035 | 401 | 2.996041666667 | 0.275208333333 | 1 | ours_nondominated_tradeoff_or_tie |
| load_035 | 503 | 4.238333333333 | 0.325833333333 | 0 | ours_nondominated_tradeoff_or_tie |
| load_065 | 101 | 8.0475 | 0.647708333333 | 1 | ours_nondominated_tradeoff_or_tie |
| load_065 | 211 | 6.565625 | 0.6375 | 0 | ours_nondominated_tradeoff_or_tie |
| load_065 | 307 | 4.024166666667 | 0.395416666667 | 1 | ours_nondominated_tradeoff_or_tie |
| load_065 | 401 | 4.106041666667 | 0.362708333333 | 0 | ours_nondominated_tradeoff_or_tie |
| load_065 | 503 | 5.377916666667 | 0.515 | 1 | ours_nondominated_tradeoff_or_tie |
| load_090 | 101 | 18.980416666667 | 1.51125 | 1 | ours_nondominated_tradeoff_or_tie |
| load_090 | 211 | 14.303541666667 | 1.296666666667 | 2 | ours_nondominated_tradeoff_or_tie |
| load_090 | 307 | 6.506041666667 | 0.582083333333 | 1 | ours_nondominated_tradeoff_or_tie |
| load_090 | 401 | 18.654166666667 | 1.35625 | 1 | ours_nondominated_tradeoff_or_tie |
| load_090 | 503 | 24.551458333333 | 1.894166666667 | 2 | ours_nondominated_tradeoff_or_tie |
| load_105 | 101 | 10.650833333333 | 0.963958333333 | 1 | ours_nondominated_tradeoff_or_tie |
| load_105 | 211 | 31.797083333333 | 2.114375 | 2 | ours_nondominated_tradeoff_or_tie |
| load_105 | 307 | 40.099791666667 | 2.561666666667 | 3 | ours_nondominated_tradeoff_or_tie |
| load_105 | 401 | 25.000625 | 1.957083333333 | 2 | ours_nondominated_tradeoff_or_tie |
| load_105 | 503 | 30.253333333333 | 2.041041666667 | 2 | ours_nondominated_tradeoff_or_tie |
| poisson_nominal | 101 | 6.555625 | 0.493958333333 | 1 | ours_nondominated_tradeoff_or_tie |
| poisson_nominal | 211 | 3.944166666667 | 0.471666666667 | 0 | ours_nondominated_tradeoff_or_tie |
| poisson_nominal | 307 | 8.318333333333 | 0.760833333333 | 1 | ours_nondominated_tradeoff_or_tie |
| poisson_nominal | 401 | 4.85875 | 0.482291666667 | 1 | ours_nondominated_tradeoff_or_tie |
| poisson_nominal | 503 | 7.090833333333 | 0.548541666667 | 1 | ours_nondominated_tradeoff_or_tie |
| static_backlog | 101 | 31.384166666667 | 1.913958333333 | 0 | ours_nondominated_tradeoff_or_tie |
| static_backlog | 211 | 31.384166666667 | 1.913958333333 | 0 | ours_nondominated_tradeoff_or_tie |
| static_backlog | 307 | 31.384166666667 | 1.913958333333 | 0 | ours_nondominated_tradeoff_or_tie |
| static_backlog | 401 | 31.384166666667 | 1.913958333333 | 0 | ours_nondominated_tradeoff_or_tie |
| static_backlog | 503 | 31.384166666667 | 1.913958333333 | 0 | ours_nondominated_tradeoff_or_tie |

## Claim Boundary

{'v1_total_cost_result_retained': True, 'metric_family_registered_after_v1_total_cost_result': True, 'arrival_tapes_observed_before_metric_freeze': False, 'known_instance_and_action_library': True, 'new_structural_instance_holdout': False, 'positive_recurrence_claim_ready': False, 'global_mmrcpsp_optimality_claim_ready': False, 'performance_superiority_claim_ready': False}
