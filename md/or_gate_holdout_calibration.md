# OR Gate: Holdout Lower-Service Calibration

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| row_count | 36 |
| selected_row_count | 6 |
| insufficient_sample_count | 3 |
| epsilon_est_all | 44.4368 |
| epsilon_est_selected_actions | 43.1457 |
| domination_violation_count | 7 |
| adjusted_eta | -43.0788 |
| lower_service_certificate_all_pass | true |
| statistical_residual_pass | false |

## Adjusted Slack Table

| Quantity | Value | Unit | Source artifact | Theorem role | Status |
|---|---:|---|---|---|---|
| Amax_i | NA | work units/slot | slot_builder | second moment bound | EMPIRICAL-OPEN |
| Smax_i | NA | work units/slot | slot_builder | second moment bound | EMPIRICAL-OPEN |
| B | 6361.354514 | work units squared | slot_builder | drift constant | PASS |
| L | 47.8979515 | service per metric | fabric_metric | fabric Lipschitz | PASS |
| rho | 0 | metric distance | fabric_metric | candidate cover radius | PASS |
| Lrho | 0 | service units/slot | fabric_metric | candidate support loss | PASS |
| epsilon_est | 43.14571657 | service units/slot | service_model | lower-service estimation loss | PASS |
| P0 | 0 | service units/slot | penalty_fit | fixed penalty constant | PASS |
| beta | 0 | service units per backlog unit | penalty_fit | queue-scaled penalty | PASS |
| alpha0 | 0 | score units | oracle_audit | fixed oracle error | PASS |
| alpha1 | 0 | score per backlog unit | oracle_audit | queue-scaled oracle error | PASS |
| delta | 0.0669218422 | service units/slot | capacity_lp | full capacity slack | PASS |
| eta | -43.07879472 | service units/slot | capacity_lp | final drift margin | FAIL |

## Selected Action Rows

| workload | profile | train n | holdout n | selected LCB/task | holdout mean/task | epsilon aggregate |
|---|---:|---:|---:|---:|---:|---:|
| cpu_heavy_local_bench | 8 | 4 | 4 | 3.3957 | 7.0311 | 29.0832 |
| gpu_heavy_jax_matmul | 1 | 1 | 1 | 34.1785 | 34.7913 | 0.612874 |
| gpu_heavy_jax_matmul | 4 | 4 | 4 | 4.64098 | 9.127 | 17.9441 |
| hybrid_rl_resac_ant | 2 | 1 | 1 | 0.166667 | 0.166667 | 0 |
| hybrid_rl_resac_ant | 3 | 2 | 1 | 0.0772593 | 0.11236 | 0.105301 |
| light_control_local | 13 | 6 | 7 | 44.3406 | 47.6595 | 43.1457 |

## Lower-Service Capacity Certificates

| taskset | profiles | delta | eta | B | theorem usable |
|---|---|---:|---:|---:|---:|
| q00_light_control | `{"light_control_local": 13}` | 123.012 | 123.012 | 310205 | true |
| q01_gpu_bound_compute | `{"gpu_heavy_jax_matmul": 8}` | 11.9469 | 11.9469 | 2925.94 | true |
| q10_cpu_host_bound | `{"cpu_heavy_local_bench": 8}` | 10.2331 | 10.2331 | 2146.7 | true |
| q11_cpu_gpu_coupled | `{"hybrid_rl_resac_ant": 3}` | 0.0659341 | 0.0659341 | 0.0891197 | true |
| hybrid_research_portfolio | `{"cpu_heavy_local_bench": 8, "gpu_heavy_jax_matmul": 4, "hybrid_rl_resac_ant": 3}` | 0.0659341 | 0.0659341 | 5492.06 | true |

## Scope

finite measured-slice holdout calibration; profiles with only one completed-history sample are reported as insufficient rather than silently treated as theorem-grade stochastic estimates. The theorem certificate uses the finite measured lower-service model directly instead of claiming stochastic generalization from sparse samples.
