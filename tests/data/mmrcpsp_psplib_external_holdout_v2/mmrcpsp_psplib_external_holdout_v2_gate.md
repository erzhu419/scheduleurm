# Disjoint post-repair PSPLIB MMRCPSP holdout

- Protocol gate: `PASS`
- Repair freeze: `c62cddd198d15faaa3d9c2d36415142a42caf727`
- Preregistration: `2856430e340906fb9f7f5470f481e322b6ee5d3c`
- Completed: `56/56`.
- Ours Pareto-nondominated: `56/56`.
- Strict all-instance baseline dominance ready: `false`.
- The 56 instances are disjoint from the v1 holdout and development fixture.
- No per-instance tuning or holdout feedback was used.

| Policy | Geo. makespan / instance best | Geo. mean flow / instance best | Pareto instances |
|---|---:|---:|---:|
| `scheduleurm_trajectory_robust_maxweight` | 1.007239 | 1.000193 | 56 |
| `shortest_processing_time` | 1.138783 | 1.099113 | 20 |
| `minimum_slack` | 1.111512 | 1.108332 | 21 |
| `greatest_rank_positional_weight` | 1.124132 | 1.098244 | 19 |
| `most_total_successors` | 1.131933 | 1.102078 | 19 |

This is prospective evidence for the registered finite holdout only.
It does not establish global MMRCPSP optimality, arbitrary-solver
dominance, or stochastic-arrival stability.
