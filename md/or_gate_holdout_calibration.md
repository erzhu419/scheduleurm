# OR Gate: Holdout Lower-Service Calibration

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| row_count | 48 |
| selected_row_count | 6 |
| insufficient_sample_count | 2 |
| epsilon_est_all | 2572.36 |
| epsilon_est_selected_actions | 1055.71 |
| domination_violation_count | 7 |
| adjusted_eta | -1055.63 |
| lower_service_certificate_all_pass | true |
| statistical_residual_pass | false |

## Adjusted Slack Table

| Quantity | Value | Unit | Source artifact | Theorem role | Status |
|---|---:|---|---|---|---|
| Amax_i | NA | work units/slot | slot_builder | second moment bound | EMPIRICAL-OPEN |
| Smax_i | NA | work units/slot | slot_builder | second moment bound | EMPIRICAL-OPEN |
| B | 34499765.49 | work units squared | slot_builder | drift constant | PASS |
| L | 5727.144174 | service per metric | fabric_metric | fabric Lipschitz | PASS |
| rho | 0 | metric distance | fabric_metric | candidate cover radius | PASS |
| Lrho | 0 | service units/slot | fabric_metric | candidate support loss | PASS |
| epsilon_est | 1055.705542 | service units/slot | service_model | lower-service estimation loss | PASS |
| P0 | 0 | service units/slot | penalty_fit | fixed penalty constant | PASS |
| beta | 0 | service units per backlog unit | penalty_fit | queue-scaled penalty | PASS |
| alpha0 | 0 | score units | oracle_audit | fixed oracle error | PASS |
| alpha1 | 0 | score per backlog unit | oracle_audit | queue-scaled oracle error | PASS |
| delta | 0.0787777101 | service units/slot | capacity_lp | full capacity slack | PASS |
| eta | -1055.626764 | service units/slot | capacity_lp | final drift margin | FAIL |

## Selected Action Rows

| workload | profile | train n | holdout n | selected LCB/task | holdout mean/task | epsilon aggregate |
|---|---:|---:|---:|---:|---:|---:|
| cpu_heavy_local_bench | 8 | 4 | 4 | 3.3957 | 7.0311 | 29.0832 |
| gpu_cnn_torch_resnet50 | 3 | 3 | 3 | 0 | 11.778 | 35.3339 |
| gpu_heavy_jax_matmul | 1 | 2 | 2 | 44.0654 | 47.7014 | 3.63594 |
| gpu_llm_distilgpt2 | 10 | 10 | 10 | 225.572 | 331.143 | 1055.71 |
| hybrid_rl_resac_ant | 2 | 4 | 4 | 0 | 0.178997 | 0.357994 |
| light_control_local | 13 | 6 | 7 | 45.7418 | 47.7457 | 26.0511 |

## Lower-Service Capacity Certificates

| taskset | profiles | delta | eta | B | theorem usable |
|---|---|---:|---:|---:|---:|
| q00_light_control | `{"light_control_local": 13}` | 123.012 | 123.012 | 310205 | true |
| q01_gpu_bound_compute | `{"gpu_heavy_jax_matmul": 1}` | 18.3409 | 18.3409 | 6895.97 | true |
| q01_gpu_bound_cnn_resnet50 | `{"gpu_cnn_torch_resnet50": 3}` | 10.8363 | 10.8363 | 2407.24 | true |
| q01_gpu_bound_llm_inference | `{"gpu_llm_distilgpt2": 10}` | 1017.03 | 1017.03 | 2.12041e+07 | true |
| q01_gpu_model_portfolio | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 10.8363 | 10.8363 | 2.12134e+07 | true |
| q10_cpu_host_bound | `{"cpu_heavy_local_bench": 8}` | 10.2331 | 10.2331 | 2146.7 | true |
| q11_cpu_gpu_coupled | `{"hybrid_rl_resac_ant": 2}` | 0.0540541 | 0.0540541 | 0.0598977 | true |
| hybrid_research_portfolio | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 0.0540541 | 0.0540541 | 2.12156e+07 | true |

## Scope

finite measured-slice holdout calibration; profiles with only one completed-history sample are reported as insufficient rather than silently treated as theorem-grade stochastic estimates. The theorem certificate uses the finite measured lower-service model directly instead of claiming stochastic generalization from sparse samples.
