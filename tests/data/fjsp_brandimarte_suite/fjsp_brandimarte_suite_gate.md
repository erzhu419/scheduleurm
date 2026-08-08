# Brandimarte Mk01-Mk15 FJSP Suite Gate

- Source-integrity and schedule-feasibility gate: `true`
- Performance-superiority claim ready: `false`
- Ours Pareto-nondominated instances: `4/15`
- Ours BKS matches: `0`
- Source commit: `e5e62a01023ff827f658255b793fd1b86e5c1707`
- Source manifest SHA-256: `6753f43a8b0033d2c3161e814bd05d49c34473e62c0e5ff0ba710bd9c1f5831c`
- License: `CC-BY-SA-4.0`
- Policy runs: `90`

BKS denotes the pinned source upper bound. A zero BKS gap means only that the heuristic matched that value; this report makes no heuristic optimality claim.

## Aggregate

| Policy | Geo. makespan/BKS | Mean makespan/BKS | Worst makespan/BKS | Geo. flow/best | BKS matches | Makespan best/tie | Flow best/tie | Pareto instances |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours: robust MaxWeight | 1.191756 | 1.197854 | 1.569948 | 1.151822 | 0 | 2 | 0 | 4 |
| Earliest Completion Time (ECT) | 1.502879 | 1.537357 | 2.384615 | 1.029731 | 0 | 0 | 6 | 10 |
| Shortest Processing Time (SPT) | 1.939552 | 1.983385 | 2.95 | 1.335679 | 0 | 0 | 0 | 0 |
| Most Work Remaining (MWKR) | 1.121654 | 1.124419 | 1.282282 | 1.155791 | 1 | 10 | 2 | 10 |
| Most Operations Remaining (MOR) | 1.158202 | 1.162602 | 1.45 | 1.113609 | 0 | 3 | 1 | 11 |
| First-In, First-Out (FIFO) | 1.659024 | 1.690695 | 2.590674 | 1.068259 | 0 | 0 | 6 | 8 |

## Ours Versus Each Baseline

| Baseline | Geo. makespan ours/base | Geo. flow ours/base | Ours Pareto-dominates | Baseline dominates ours | Tradeoff/equal |
|---|---:|---:|---:|---:|---:|
| Earliest Completion Time (ECT) | 0.792982 | 1.118566 | 2 | 1 | 12 |
| Shortest Processing Time (SPT) | 0.614449 | 0.862349 | 12 | 0 | 3 |
| Most Work Remaining (MWKR) | 1.062499 | 0.996565 | 1 | 7 | 7 |
| Most Operations Remaining (MOR) | 1.028971 | 1.034314 | 1 | 11 | 3 |
| First-In, First-Out (FIFO) | 0.718348 | 1.078223 | 5 | 0 | 10 |

## Per-Instance Summary

| Instance | Size | LB | BKS | Status | Ours makespan | Ours BKS gap | Ours mean flow | Best makespan | Best mean flow | Pareto frontier |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| `mk01` | 10x6 | 40 | 40 | `closed` | 45 | 12.50% | 34.4 | 44 (most_work_remaining) | 30.9 (earliest_completion_time) | ours_robust_maxweight, earliest_completion_time, most_work_remaining, most_operations_remaining |
| `mk02` | 10x6 | 26 | 26 | `closed` | 34 | 30.77% | 25.9 | 29 (most_work_remaining) | 25 (fifo_job_order) | ours_robust_maxweight, most_work_remaining, most_operations_remaining, fifo_job_order |
| `mk03` | 15x8 | 204 | 204 | `closed` | 234 | 14.71% | 160.4 | 204 (most_work_remaining) | 148.666667 (most_operations_remaining) | most_work_remaining, most_operations_remaining |
| `mk04` | 15x8 | 60 | 60 | `closed` | 74 | 23.33% | 57.6 | 74 (ours_robust_maxweight) | 50.2 (earliest_completion_time) | ours_robust_maxweight, earliest_completion_time |
| `mk05` | 15x4 | 172 | 172 | `closed` | 185 | 7.56% | 174.4 | 185 (ours_robust_maxweight) | 130.6 (earliest_completion_time) | ours_robust_maxweight, earliest_completion_time, most_operations_remaining, fifo_job_order |
| `mk06` | 10x15 | 57 | 57 | `closed` | 72 | 26.32% | 67 | 67 (most_work_remaining) | 63.8 (most_work_remaining) | most_work_remaining |
| `mk07` | 20x5 | 139 | 139 | `closed` | 170 | 22.30% | 125.95 | 154 (most_operations_remaining) | 109 (fifo_job_order) | most_operations_remaining, fifo_job_order |
| `mk08` | 20x10 | 523 | 523 | `closed` | 563 | 7.65% | 470.55 | 533 (most_work_remaining) | 386.05 (fifo_job_order) | earliest_completion_time, most_work_remaining, most_operations_remaining, fifo_job_order |
| `mk09` | 20x10 | 307 | 307 | `closed` | 356 | 15.96% | 322.9 | 331 (most_work_remaining) | 281.85 (earliest_completion_time) | earliest_completion_time, most_work_remaining, fifo_job_order |
| `mk10` | 20x15 | 189 | 193 | `open` | 303 | 56.99% | 242.5 | 230 (most_work_remaining) | 219.1 (most_work_remaining) | most_work_remaining |
| `mk11` | 30x5 | 609 | 609 | `closed` | 683 | 12.15% | 595.6 | 660 (most_operations_remaining) | 433.866667 (fifo_job_order) | earliest_completion_time, most_operations_remaining, fifo_job_order |
| `mk12` | 30x10 | 508 | 508 | `closed` | 570 | 12.20% | 442.633333 | 562 (most_work_remaining) | 344.533333 (earliest_completion_time) | earliest_completion_time, most_work_remaining, most_operations_remaining |
| `mk13` | 30x10 | 381 | 390 | `open` | 496 | 27.18% | 392.666667 | 469 (most_work_remaining) | 346 (fifo_job_order) | earliest_completion_time, most_work_remaining, most_operations_remaining, fifo_job_order |
| `mk14` | 30x15 | 694 | 694 | `closed` | 722 | 4.03% | 557.866667 | 719 (most_work_remaining) | 541.2 (earliest_completion_time) | earliest_completion_time, most_work_remaining, most_operations_remaining |
| `mk15` | 30x15 | 333 | 333 | `closed` | 410 | 23.12% | 357.433333 | 402 (most_operations_remaining) | 316.466667 (fifo_job_order) | earliest_completion_time, most_operations_remaining, fifo_job_order |

## Every Policy On Every Instance

| Instance | Policy | Makespan | BKS gap | LB gap | Mean flow | Matched BKS | Pareto | Feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mk01` | Ours: robust MaxWeight | 45 | 12.50% | 12.50% | 34.4 | false | true | true |
| `mk01` | Earliest Completion Time (ECT) | 56 | 40.00% | 40.00% | 30.9 | false | true | true |
| `mk01` | Shortest Processing Time (SPT) | 71 | 77.50% | 77.50% | 39.6 | false | false | true |
| `mk01` | Most Work Remaining (MWKR) | 44 | 10.00% | 10.00% | 35.4 | false | true | true |
| `mk01` | Most Operations Remaining (MOR) | 47 | 17.50% | 17.50% | 33.4 | false | true | true |
| `mk01` | First-In, First-Out (FIFO) | 69 | 72.50% | 72.50% | 37.7 | false | false | true |
| `mk02` | Ours: robust MaxWeight | 34 | 30.77% | 30.77% | 25.9 | false | true | true |
| `mk02` | Earliest Completion Time (ECT) | 62 | 138.46% | 138.46% | 27.9 | false | false | true |
| `mk02` | Shortest Processing Time (SPT) | 65 | 150.00% | 150.00% | 30.2 | false | false | true |
| `mk02` | Most Work Remaining (MWKR) | 29 | 11.54% | 11.54% | 27.2 | false | true | true |
| `mk02` | Most Operations Remaining (MOR) | 32 | 23.08% | 23.08% | 26.3 | false | true | true |
| `mk02` | First-In, First-Out (FIFO) | 50 | 92.31% | 92.31% | 25 | false | true | true |
| `mk03` | Ours: robust MaxWeight | 234 | 14.71% | 14.71% | 160.4 | false | false | true |
| `mk03` | Earliest Completion Time (ECT) | 322 | 57.84% | 57.84% | 176 | false | false | true |
| `mk03` | Shortest Processing Time (SPT) | 405 | 98.53% | 98.53% | 222.866667 | false | false | true |
| `mk03` | Most Work Remaining (MWKR) | 204 | 0.00% | 0.00% | 157.533333 | true | true | true |
| `mk03` | Most Operations Remaining (MOR) | 222 | 8.82% | 8.82% | 148.666667 | false | true | true |
| `mk03` | First-In, First-Out (FIFO) | 369 | 80.88% | 80.88% | 182.133333 | false | false | true |
| `mk04` | Ours: robust MaxWeight | 74 | 23.33% | 23.33% | 57.6 | false | true | true |
| `mk04` | Earliest Completion Time (ECT) | 91 | 51.67% | 51.67% | 50.2 | false | true | true |
| `mk04` | Shortest Processing Time (SPT) | 177 | 195.00% | 195.00% | 84.666667 | false | false | true |
| `mk04` | Most Work Remaining (MWKR) | 75 | 25.00% | 25.00% | 62.133333 | false | false | true |
| `mk04` | Most Operations Remaining (MOR) | 87 | 45.00% | 45.00% | 60.466667 | false | false | true |
| `mk04` | First-In, First-Out (FIFO) | 134 | 123.33% | 123.33% | 63.6 | false | false | true |
| `mk05` | Ours: robust MaxWeight | 185 | 7.56% | 7.56% | 174.4 | false | true | true |
| `mk05` | Earliest Completion Time (ECT) | 237 | 37.79% | 37.79% | 130.6 | false | true | true |
| `mk05` | Shortest Processing Time (SPT) | 260 | 51.16% | 51.16% | 146.2 | false | false | true |
| `mk05` | Most Work Remaining (MWKR) | 187 | 8.72% | 8.72% | 171.866667 | false | false | true |
| `mk05` | Most Operations Remaining (MOR) | 187 | 8.72% | 8.72% | 167.133333 | false | true | true |
| `mk05` | First-In, First-Out (FIFO) | 230 | 33.72% | 33.72% | 134.933333 | false | true | true |
| `mk06` | Ours: robust MaxWeight | 72 | 26.32% | 26.32% | 67 | false | false | true |
| `mk06` | Earliest Completion Time (ECT) | 88 | 54.39% | 54.39% | 64.8 | false | false | true |
| `mk06` | Shortest Processing Time (SPT) | 114 | 100.00% | 100.00% | 93.2 | false | false | true |
| `mk06` | Most Work Remaining (MWKR) | 67 | 17.54% | 17.54% | 63.8 | false | true | true |
| `mk06` | Most Operations Remaining (MOR) | 71 | 24.56% | 24.56% | 65.1 | false | false | true |
| `mk06` | First-In, First-Out (FIFO) | 96 | 68.42% | 68.42% | 63.9 | false | false | true |
| `mk07` | Ours: robust MaxWeight | 170 | 22.30% | 22.30% | 125.95 | false | false | true |
| `mk07` | Earliest Completion Time (ECT) | 207 | 48.92% | 48.92% | 110.1 | false | false | true |
| `mk07` | Shortest Processing Time (SPT) | 261 | 87.77% | 87.77% | 143.7 | false | false | true |
| `mk07` | Most Work Remaining (MWKR) | 158 | 13.67% | 13.67% | 142.4 | false | false | true |
| `mk07` | Most Operations Remaining (MOR) | 154 | 10.79% | 10.79% | 120.65 | false | true | true |
| `mk07` | First-In, First-Out (FIFO) | 204 | 46.76% | 46.76% | 109 | false | true | true |
| `mk08` | Ours: robust MaxWeight | 563 | 7.65% | 7.65% | 470.55 | false | false | true |
| `mk08` | Earliest Completion Time (ECT) | 648 | 23.90% | 23.90% | 390.6 | false | true | true |
| `mk08` | Shortest Processing Time (SPT) | 739 | 41.30% | 41.30% | 446.25 | false | false | true |
| `mk08` | Most Work Remaining (MWKR) | 533 | 1.91% | 1.91% | 462.2 | false | true | true |
| `mk08` | Most Operations Remaining (MOR) | 536 | 2.49% | 2.49% | 448.7 | false | true | true |
| `mk08` | First-In, First-Out (FIFO) | 676 | 29.25% | 29.25% | 386.05 | false | true | true |
| `mk09` | Ours: robust MaxWeight | 356 | 15.96% | 15.96% | 322.9 | false | false | true |
| `mk09` | Earliest Completion Time (ECT) | 436 | 42.02% | 42.02% | 281.85 | false | true | true |
| `mk09` | Shortest Processing Time (SPT) | 507 | 65.15% | 65.15% | 327.25 | false | false | true |
| `mk09` | Most Work Remaining (MWKR) | 331 | 7.82% | 7.82% | 309.2 | false | true | true |
| `mk09` | Most Operations Remaining (MOR) | 347 | 13.03% | 13.03% | 312.5 | false | false | true |
| `mk09` | First-In, First-Out (FIFO) | 431 | 40.39% | 40.39% | 287.6 | false | true | true |
| `mk10` | Ours: robust MaxWeight | 303 | 56.99% | 60.32% | 242.5 | false | false | true |
| `mk10` | Earliest Completion Time (ECT) | 439 | 127.46% | 132.28% | 235.1 | false | false | true |
| `mk10` | Shortest Processing Time (SPT) | 532 | 175.65% | 181.48% | 295.8 | false | false | true |
| `mk10` | Most Work Remaining (MWKR) | 230 | 19.17% | 21.69% | 219.1 | false | true | true |
| `mk10` | Most Operations Remaining (MOR) | 238 | 23.32% | 25.93% | 222.8 | false | false | true |
| `mk10` | First-In, First-Out (FIFO) | 500 | 159.07% | 164.55% | 242.7 | false | false | true |
| `mk11` | Ours: robust MaxWeight | 683 | 12.15% | 12.15% | 595.6 | false | false | true |
| `mk11` | Earliest Completion Time (ECT) | 675 | 10.84% | 10.84% | 434 | false | true | true |
| `mk11` | Shortest Processing Time (SPT) | 1012 | 66.17% | 66.17% | 563 | false | false | true |
| `mk11` | Most Work Remaining (MWKR) | 661 | 8.54% | 8.54% | 584.333333 | false | false | true |
| `mk11` | Most Operations Remaining (MOR) | 660 | 8.37% | 8.37% | 559.966667 | false | true | true |
| `mk11` | First-In, First-Out (FIFO) | 860 | 41.22% | 41.22% | 433.866667 | false | true | true |
| `mk12` | Ours: robust MaxWeight | 570 | 12.20% | 12.20% | 442.633333 | false | false | true |
| `mk12` | Earliest Completion Time (ECT) | 621 | 22.24% | 22.24% | 344.533333 | false | true | true |
| `mk12` | Shortest Processing Time (SPT) | 878 | 72.83% | 72.83% | 460.566667 | false | false | true |
| `mk12` | Most Work Remaining (MWKR) | 562 | 10.63% | 10.63% | 460.166667 | false | true | true |
| `mk12` | Most Operations Remaining (MOR) | 566 | 11.42% | 11.42% | 430 | false | true | true |
| `mk12` | First-In, First-Out (FIFO) | 742 | 46.06% | 46.06% | 398.766667 | false | false | true |
| `mk13` | Ours: robust MaxWeight | 496 | 27.18% | 30.18% | 392.666667 | false | false | true |
| `mk13` | Earliest Completion Time (ECT) | 581 | 48.97% | 52.49% | 349 | false | true | true |
| `mk13` | Shortest Processing Time (SPT) | 753 | 93.08% | 97.64% | 464.8 | false | false | true |
| `mk13` | Most Work Remaining (MWKR) | 469 | 20.26% | 23.10% | 402.1 | false | true | true |
| `mk13` | Most Operations Remaining (MOR) | 476 | 22.05% | 24.93% | 379.733333 | false | true | true |
| `mk13` | First-In, First-Out (FIFO) | 611 | 56.67% | 60.37% | 346 | false | true | true |
| `mk14` | Ours: robust MaxWeight | 722 | 4.03% | 4.03% | 557.866667 | false | false | true |
| `mk14` | Earliest Completion Time (ECT) | 863 | 24.35% | 24.35% | 541.2 | false | true | true |
| `mk14` | Shortest Processing Time (SPT) | 1184 | 70.61% | 70.61% | 724.466667 | false | false | true |
| `mk14` | Most Work Remaining (MWKR) | 719 | 3.60% | 3.60% | 552.066667 | false | true | true |
| `mk14` | Most Operations Remaining (MOR) | 722 | 4.03% | 4.03% | 541.866667 | false | true | true |
| `mk14` | First-In, First-Out (FIFO) | 1072 | 54.47% | 54.47% | 568.4 | false | false | true |
| `mk15` | Ours: robust MaxWeight | 410 | 23.12% | 23.12% | 357.433333 | false | false | true |
| `mk15` | Earliest Completion Time (ECT) | 590 | 77.18% | 77.18% | 330.766667 | false | true | true |
| `mk15` | Shortest Processing Time (SPT) | 767 | 130.33% | 130.33% | 509.766667 | false | false | true |
| `mk15` | Most Work Remaining (MWKR) | 427 | 28.23% | 28.23% | 352.733333 | false | false | true |
| `mk15` | Most Operations Remaining (MOR) | 402 | 20.72% | 20.72% | 346.2 | false | true | true |
| `mk15` | First-In, First-Out (FIFO) | 636 | 90.99% | 90.99% | 316.466667 | false | true | true |

## Scope

Deterministic constructive-heuristic evidence on the hash-locked Brandimarte Mk01-Mk15 suite.  BKS matches are equality to the pinned source upper bound, not an independent optimality certificate.
