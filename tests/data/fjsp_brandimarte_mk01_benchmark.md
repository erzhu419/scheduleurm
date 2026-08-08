# Flexible Job-Shop Scheduling Problem Benchmark

- Instance: `fjsp_brandimarte_mk01`
- Source SHA-256: `a500841f969eed4e730688c71dd41aeff80b49b3d0e2ec4c408a91904225556b`
- Size: 10 jobs, 6 machines, 55 operations
- Constructive lower bound: `26`
- Declared best-known makespan: `40`
- Feasibility gate: `true`

| Policy | Makespan | Makespan/BKS | Mean flow | Reassignments | Reconfiguration time | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| Ours: robust MaxWeight | 45 | 1.125 | 34.4 | 0 | 0 | true |
| Earliest Completion Time (ECT) | 56 | 1.4 | 30.9 | 0 | 0 | true |
| Shortest Processing Time (SPT) | 71 | 1.775 | 39.6 | 0 | 0 | true |
| Most Work Remaining (MWKR) | 44 | 1.1 | 35.4 | 0 | 0 | true |
| Most Operations Remaining (MOR) | 47 | 1.175 | 33.4 | 0 | 0 | true |
| First-In, First-Out (FIFO) | 69 | 1.725 | 37.7 | 0 | 0 | true |

## Action And Reassignment Semantics

Every decision considers the next precedence-feasible operation of each unfinished job on every machine listed for that operation.

The reassignment extension is `false`. Its scope is `disabled` and its declared setup time is `0`.
Standard FJSPLIB semantics are non-preemptive; running-operation migration is not claimed.

## Comparison Boundary

Deterministic benchmark evidence on the declared FJSP instance.  This is not a claim of optimality, arbitrary-instance dominance, shop-floor deployment, or running-operation migration.
