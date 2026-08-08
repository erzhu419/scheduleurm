# FJSP trajectory/configuration-action upgrade

- Gate: `PASS`
- Frozen configuration: `b9fa529ff6218035d4fe0173b9faf09909db96fdd8b24035d8e52a292359fe07`
- Internal diagnostic partition: Mk01-Mk05 development; Mk06-Mk10 calibration; Mk11-Mk15 evaluation.
- Prospective external holdout: not yet run from a repository-frozen configuration.
- Exactness: exact maximizer of the generated finite trajectory family; not all FJSP schedules.
- Ours Pareto-nondominated: `15/15`.
- Strict all-instance classic-policy dominance ready: `false`.

## Fixed theorem mapping

For common frame `H`, each job has lower-service `(H-C_j)/H`, and the system
coordinate has `(H-C_max)/H`. The normalized queue-weighted service is scored
minus a latency penalty bounded by the single frozen coefficient. The family
oracle gap is zero; approximation to the full FJSP trajectory space is not claimed.

## Aggregate comparison

| Policy | Geo. makespan/BKS | Geo. flow/best | Pareto instances |
|---|---:|---:|---:|
| Ours: trajectory robust MaxWeight | 1.118905 | 1.081916 | 15 |
| Ours: robust MaxWeight | 1.191756 | 1.152279 | 3 |
| Earliest Completion Time (ECT) | 1.502879 | 1.030139 | 10 |
| Shortest Processing Time (SPT) | 1.939552 | 1.336209 | 0 |
| Most Work Remaining (MWKR) | 1.121654 | 1.156250 | 6 |
| Most Operations Remaining (MOR) | 1.158202 | 1.114051 | 9 |
| First-In, First-Out (FIFO) | 1.659024 | 1.068683 | 8 |

## Instance results

| Split | Instance | Family | Runtime (s) | Ours makespan | Ours mean flow | Pareto frontier |
|---|---|---:|---:|---:|---:|---|
| development | mk01 | 22 | 0.367488 | 43 | 35.100000 | ours_trajectory_configuration, ours_robust_maxweight, earliest_completion_time, most_operations_remaining |
| development | mk02 | 26 | 0.681775 | 29 | 27.200000 | ours_trajectory_configuration, ours_robust_maxweight, most_work_remaining, most_operations_remaining, fifo_job_order |
| development | mk03 | 27 | 2.612973 | 204 | 157.533333 | ours_trajectory_configuration, most_work_remaining, most_operations_remaining |
| development | mk04 | 34 | 0.969759 | 74 | 57.600000 | ours_trajectory_configuration, ours_robust_maxweight, earliest_completion_time |
| development | mk05 | 42 | 0.991866 | 184 | 170.133333 | ours_trajectory_configuration, earliest_completion_time, most_operations_remaining, fifo_job_order |
| calibration | mk06 | 26 | 1.789786 | 67 | 63.800000 | ours_trajectory_configuration, most_work_remaining |
| calibration | mk07 | 47 | 1.735938 | 154 | 120.650000 | ours_trajectory_configuration, most_operations_remaining, fifo_job_order |
| calibration | mk08 | 33 | 3.350480 | 531 | 443.700000 | ours_trajectory_configuration, earliest_completion_time, fifo_job_order |
| calibration | mk09 | 27 | 4.495219 | 331 | 309.200000 | ours_trajectory_configuration, earliest_completion_time, most_work_remaining, fifo_job_order |
| calibration | mk10 | 36 | 4.389892 | 228 | 217.800000 | ours_trajectory_configuration |
| holdout | mk11 | 42 | 3.291219 | 675 | 434 | ours_trajectory_configuration, earliest_completion_time, most_operations_remaining, fifo_job_order |
| holdout | mk12 | 36 | 2.668745 | 621 | 344.533333 | ours_trajectory_configuration, earliest_completion_time, most_work_remaining, most_operations_remaining |
| holdout | mk13 | 25 | 6.196292 | 464 | 376.566667 | ours_trajectory_configuration, earliest_completion_time, fifo_job_order |
| holdout | mk14 | 19 | 5.277504 | 722 | 541.866667 | ours_trajectory_configuration, earliest_completion_time, most_work_remaining, most_operations_remaining |
| holdout | mk15 | 29 | 7.399889 | 402 | 346.200000 | ours_trajectory_configuration, earliest_completion_time, most_operations_remaining, fifo_job_order |

## Claim boundary

This is deterministic, hash-locked, finite-family FJSP evidence. CP-SAT is not
used. A zero oracle gap refers only to the generated trajectory family. The
artifact does not claim global FJSP optimality or stochastic throughput stability.
