# Flexible Job-Shop Scheduling Problem Benchmark

- Instance: `fjsp_brandimarte_tiny`
- Source SHA-256: `4ae7834cd8e2dc8ddd89bec9a68f4b1b9ea36c97ac4919026bce69ca61ac1b2f`
- Size: 4 jobs, 3 machines, 8 operations
- Constructive lower bound: `6`
- Declared best-known makespan: `not supplied`
- Feasibility gate: `true`

| Policy | Makespan | Makespan/BKS | Mean flow | Reassignments | Reconfiguration time | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| Ours: robust MaxWeight | 9 | n/a | 8 | 2 | 2 | true |
| Earliest Completion Time (ECT) | 13 | n/a | 8 | 1 | 1 | true |
| Shortest Processing Time (SPT) | 14 | n/a | 8.25 | 0 | 0 | true |
| Most Work Remaining (MWKR) | 9 | n/a | 7.75 | 2 | 2 | true |
| Most Operations Remaining (MOR) | 10 | n/a | 8.5 | 2 | 2 | true |
| First-In, First-Out (FIFO) | 13 | n/a | 8 | 1 | 1 | true |

## Action And Reassignment Semantics

Every decision considers the next precedence-feasible operation of each unfinished job on every machine listed for that operation.

The reassignment extension is `true`. Its scope is `unstarted_operation_only` and its declared setup time is `1`.
Standard FJSPLIB semantics are non-preemptive; running-operation migration is not claimed.

## Comparison Boundary

Deterministic benchmark evidence on the declared FJSP instance.  This is not a claim of optimality, arbitrary-instance dominance, shop-floor deployment, or running-operation migration.
