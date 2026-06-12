# q00/q10 Generalization Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `q00_declared_local_bucket_closed` | true |
| `q10_declared_local_bucket_closed` | true |
| `remote_cpu_has_multi_profile_evidence` | true |
| `remote_cpu_broad_theorem_ready` | true |
| `broad_q00_q10_generalization_ready` | false |

## Local Buckets

| Bucket | Positive profiles | Boundary profiles | Node buckets |
|---|---|---|---|
| `q00:light_control_local` | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` | `[14, 16]` | `['local:cpu']` |
| `q10:cpu_heavy_local_bench` | `[1, 2, 3, 4, 5, 6, 7, 8, 9]` | `[10]` | `['local:cpu']` |

## Remote CPU Evidence

| Workload | Kind | Node buckets | Positive profiles | Boundary profiles | Source kinds |
|---|---|---|---|---|---|
| `assumption_agent_live_benchmark_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `assumption_agent_meta_qa_evolution_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `assumption_agent_performance_validation_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `assumption_agent_phase2_v20_framework_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `assumption_agent_unittest_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_cpu_training_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_diagnostic_shard_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_diagnostic_shard_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_diagnostic_shard_c9_16_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_diagnostic_shard_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_mujoco_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_mujoco_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_mujoco_c9_16_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_mujoco_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_mujoco_policy_union_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_train_compare_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_train_compare_c9_16_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bamor_train_compare_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `bapr_id_ood_merge_cpu_eval_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_cpu_eval_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_env_validation_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_feed_conversion_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_policy_rollout_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_pytest_cpu_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_pytest_sumo_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_snapshot_generation_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_snapshot_generation_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_sumo_generation_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_traffic_signal_phase1_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cfcmt_traffic_signal_phase2_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `cpu_heavy_local_fabric_completed_history` | `cpu_sumo_transit` | `['production-history:cpu-fabric']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_baseline_rule_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_cpu_ablation_c17_32` | `cpu_sumo_transit` | `['jtl110cpu2:cpu128']` | `[1, 2, 4]` | `[8]` | `['measured_curve']` |
| `freqduet_cpu_ablation_c33_64_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_cpu_ablation_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_cpu_ablation_c65p_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_cpu_ablation_c9_16` | `cpu_sumo_transit` | `['jtl110cpu2:cpu128']` | `[1, 2, 4, 8]` | `[]` | `['measured_curve']` |
| `freqduet_cpu_ablation_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_paper_longtrain_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_preflight_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_promoted_ep100_c65p_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `cpu_sumo_transit` | `['jtl110cpu2:cpu128']` | `[1, 2, 4, 8]` | `[]` | `['measured_curve']` |
| `freqduet_runner_v3_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_runner_v3_c33_64_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_runner_v3_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_runner_v3_c9_16_residual_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_runner_v3_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `freqduet_spacectx_ep100_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `h2oplus_shell_eval_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `nature_emissions_extract_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `nature_emissions_routeguard_analysis_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `nature_emissions_sumo_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `offline_sumo_eval_c33_64_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `offline_sumo_eval_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `resac_bus_seed_extension_cpu_eval_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `resac_conda_pack_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `resco_config_eval_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `scheduleurm_control_plane_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `scheduleurm_hpc_relay_smoke_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `sensing_pems_cache_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `sensing_voltage_cache_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_demand_estimator_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_freqhrl_analysis_matrix_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_freqhrl_import_smoke_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_freqhrl_merge_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_freqhrl_tests_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_freqhrl_tests_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_gap_closure_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_control_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_merge_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c17_32_residual_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c33_64_batch_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c33_64_single_seed_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c3_8_persistent_stress_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c65p_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c9_16_bounded_wait_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c9_16_residual_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c9_16_wait_credit_shell_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_promotion_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_real_demand_alighting_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_real_demand_batch_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_real_demand_batch_c9_16_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_native_real_demand_safe_wait_c9_16_profile_extension` | `cpu_sumo_transit` | `['production-history:cpu-fabric']` | `[1]` | `[]` | `['completed_history']` |
| `transit_surrogate_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_surrogate_validation_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_policy_c33_64_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_policy_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_policy_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_pressure_matrix_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_pressure_matrix_c33_64_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_pressure_matrix_c9_16_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_pressure_merge_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_promotion_recovery_c17_32_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_public_csv_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `transit_trading_sweep_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `zsw_m21_sumo_eval_c3_8_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `zsw_metrics_parser_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `cpu_sumo_transit` | `['production-history:cpu']` | `[1]` | `[]` | `['completed_history']` |

## Scope

Evidence gate for q00/q10 scope. Declared local q00/q10 buckets are closed; remote CPU/SUMO production evidence is inventoried and includes multi-profile remote CPU data where present. Broad all-CPU/data-loader generalization remains false until remote curves include comparable capacity boundaries and admission rules.
