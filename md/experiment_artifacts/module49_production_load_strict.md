# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = False
record_count_window = 5980
mapped_task_count = 2573
representative_mapped_task_count = 0
unmapped_task_count = 3407
mapped_fraction = 0.4302675585284281
```

| Workload | Count | Lambda |
|---|---:|---:|
| `bamor_diagnostic_shard_c3_8_completed_history` | 25 | 6.442901235 |
| `bamor_diagnostic_shard_c9_16_completed_history` | 34 | 16.666666667 |
| `bamor_mujoco_c3_8_completed_history` | 166 | 3.202160494 |
| `bamor_mujoco_c9_16_completed_history` | 3 | 0.006944444 |
| `bamor_train_compare_c3_8_completed_history` | 68 | 2.165123457 |
| `bamor_train_compare_c9_16_completed_history` | 6 | 0.001273148 |
| `cfcmt_env_validation_c_le2_completed_history` | 5 | 0.033641975 |
| `cfcmt_feed_conversion_c_le2_completed_history` | 5 | 0.000001929 |
| `cfcmt_policy_rollout_c_le2_completed_history` | 10 | 0.054328704 |
| `cfcmt_snapshot_generation_c_le2_completed_history` | 5 | 0.000601852 |
| `cfcmt_sumo_generation_c_le2_completed_history` | 4 | 0.040972222 |
| `cfcmt_traffic_signal_phase2_c_le2_completed_history` | 1 | 0.000000386 |
| `cpu_heavy_local_bench` | 55 | 0.021219136 |
| `freqduet_baseline_rule_c_le2_completed_history` | 5 | 0.000038580 |
| `freqduet_cpu_ablation_c17_32` | 190 | 0.005277778 |
| `freqduet_cpu_ablation_c33_64_completed_history` | 63 | 0.060239198 |
| `freqduet_cpu_ablation_c3_8_completed_history` | 42 | 0.006945602 |
| `freqduet_cpu_ablation_c65p_completed_history` | 11 | 0.012048611 |
| `freqduet_cpu_ablation_c9_16` | 135 | 0.074498457 |
| `freqduet_cpu_ablation_c_le2_completed_history` | 7 | 0.000928627 |
| `freqduet_paper_longtrain_c17_32_completed_history` | 35 | 0.000013503 |
| `freqduet_preflight_c_le2_completed_history` | 14 | 0.000005401 |
| `freqduet_promoted_ep100_c65p_completed_history` | 6 | 0.023148148 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |
| `freqduet_runner_v3_c17_32_completed_history` | 16 | 0.000216049 |
| `freqduet_runner_v3_c3_8_completed_history` | 86 | 0.001045139 |
| `freqduet_runner_v3_c9_16_residual_completed_history` | 13 | 0.000185185 |
| `freqduet_runner_v3_c_le2_completed_history` | 85 | 0.001798611 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 476 | 0.014691358 |
| `light_control_local` | 206 | 0.794753086 |
| `sumo_eval_simple_sac_c_le2` | 71 | 0.000027392 |
| `transit_freqhrl_analysis_matrix_c_le2_completed_history` | 14 | 0.000005401 |
| `transit_freqhrl_import_smoke_c_le2_completed_history` | 2 | 0.000000772 |
| `transit_freqhrl_merge_c_le2_completed_history` | 8 | 0.000003086 |
| `transit_native_control_c_le2_completed_history` | 9 | 0.000069444 |
| `transit_native_promotion_c17_32_residual_completed_history` | 16 | 0.000505787 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | 153 | 0.017438657 |
| `transit_native_promotion_c33_64_batch_completed_history` | 179 | 0.019272377 |
| `transit_native_promotion_c65p_completed_history` | 115 | 0.009266204 |
| `transit_native_promotion_c9_16_bounded_wait_completed_history` | 9 | 0.000900463 |
| `transit_native_promotion_c9_16_residual_completed_history` | 45 | 0.003028935 |
| `transit_native_promotion_c_le2_completed_history` | 19 | 0.000201389 |
| `transit_surrogate_validation_c_le2_completed_history` | 3 | 0.019037037 |
| `transit_trading_policy_c_le2_completed_history` | 18 | 1.600000000 |
| `transit_trading_sweep_c_le2_completed_history` | 14 | 16.295833333 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | 50 | 0.347222222 |

| Quantity | Value |
|---|---:|
| `delta` | 0.000050804 |
| `mapped_capacity_usable_for_theorem` | true |
| `global_coverage_usable_for_theorem` | false |
| `usable_for_global_theorem` | false |

## Unmapped Reasons

| Reason | Count |
|---|---:|
| `scheduleurmbench_unknown` | 155 |
| `unmapped_cpu` | 1661 |
| `unmapped_gpu` | 1591 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
