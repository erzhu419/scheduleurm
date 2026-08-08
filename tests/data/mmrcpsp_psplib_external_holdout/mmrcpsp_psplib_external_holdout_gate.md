# Post-freeze PSPLIB MMRCPSP holdout

- Protocol gate: `FAIL`
- Frozen policy commit: `61b835eb17216ccd2e5e950d2aceab09e05a3f27`
- Preregistration commit: `5d806861d074b9957eac9878e5f78cb70f595ad8`
- Registered official instances: `56`
- The instance selection was outcome-blind and committed before materialization.
- No per-instance tuning or holdout feedback was used.
- Completed before fail-closed stop: `0/56`.
- Ours Pareto-nondominated among completed rows: `0/0`.
- Strict all-instance baseline dominance ready: `false`.
- First failure: `j102_10` / `TrajectoryUpgradeError`.
- Failure reason: `j102_10: rollout seed scheduleurm_robust_maxweight is infeasible: resource_deadlock:eligible=[6, 9]:nonrenewable_consumed=[27, 20]`

| Policy | Geo. makespan / instance best | Geo. mean flow / instance best | Pareto instances |
|---|---:|---:|---:|
| `scheduleurm_trajectory_robust_maxweight` | n/a | n/a | 0 |
| `shortest_processing_time` | n/a | n/a | 0 |
| `minimum_slack` | n/a | n/a | 0 |
| `greatest_rank_positional_weight` | n/a | n/a | 0 |
| `most_total_successors` | n/a | n/a | 0 |

This gate establishes prospective behavior for the preregistered finite
PSPLIB holdout. It does not claim global MMRCPSP optimality, dominance
over arbitrary solvers or instances, or stochastic-arrival stability.
