# Post-freeze external FJSP holdout

- Protocol gate: `PASS`
- Frozen repository commit: `9a922be91161829b714e7c1c11b69d66e799f203`
- Frozen config: `b9fa529ff6218035d4fe0173b9faf09909db96fdd8b24035d8e52a292359fe07`
- External suite: complete Kacem1-Kacem4 collection, absent from the freeze commit.
- No per-instance tuning or holdout feedback was used.
- Ours Pareto-nondominated: `4/4`.
- Strict all-instance classic-policy dominance ready: `false`.

| Policy | Geo. makespan/BKS | Geo. flow/best | Pareto instances |
|---|---:|---:|---:|
| Ours: trajectory robust MaxWeight | 1.044466 | 1.076281 | 4 |
| Ours: robust MaxWeight | 1.305217 | 1.144046 | 1 |
| Earliest Completion Time (ECT) | 1.311209 | 1.010029 | 2 |
| Shortest Processing Time (SPT) | 1.494377 | 1.075659 | 0 |
| Most Work Remaining (MWKR) | 1.125978 | 1.166320 | 0 |
| Most Operations Remaining (MOR) | 1.145037 | 1.054381 | 3 |
| First-In, First-Out (FIFO) | 1.266030 | 1.024917 | 2 |

| Instance | Ours makespan | Ours mean flow | BKS | Pareto frontier |
|---|---:|---:|---:|---|
| kacem1 | 11 | 8.5 | 11 | ours_trajectory_configuration, most_operations_remaining, fifo_job_order |
| kacem2 | 12 | 9.6 | 11 | ours_trajectory_configuration, ours_robust_maxweight, most_operations_remaining |
| kacem3 | 7 | 5.6 | 7 | ours_trajectory_configuration, earliest_completion_time, most_operations_remaining, fifo_job_order |
| kacem4 | 12 | 10 | 11 | ours_trajectory_configuration, earliest_completion_time |

The gate establishes prospective post-freeze behavior for this external finite
suite. It does not claim global FJSP optimality, arbitrary-instance dominance,
or stochastic-arrival throughput optimality.
