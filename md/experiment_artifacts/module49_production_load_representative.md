# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = True
record_count_window = 5877
mapped_task_count = 3927
representative_mapped_task_count = 1889
unmapped_task_count = 1950
mapped_fraction = 0.6681980602348137
```

| Workload | Count | Lambda |
|---|---:|---:|
| `bamor_diagnostic_shard_c3_8_completed_history` | 25 | 6.442901235 |
| `bamor_mujoco_c3_8_completed_history` | 115 | 2.218364198 |
| `bamor_train_compare_c3_8_completed_history` | 68 | 2.165123457 |
| `cpu_heavy_local_bench` | 379 | 0.146219136 |
| `freqduet_cpu_ablation_c17_32` | 190 | 0.005277778 |
| `freqduet_cpu_ablation_c33_64_completed_history` | 63 | 0.060239198 |
| `freqduet_cpu_ablation_c3_8_completed_history` | 42 | 0.006945602 |
| `freqduet_cpu_ablation_c9_16` | 129 | 0.068325617 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |
| `freqduet_runner_v3_c3_8_completed_history` | 86 | 0.001045139 |
| `freqduet_runner_v3_c_le2_completed_history` | 84 | 0.001798225 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 2041 | 0.062993827 |
| `light_control_local` | 206 | 0.794753086 |
| `sumo_eval_simple_sac_c_le2` | 71 | 0.000027392 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | 140 | 0.015726080 |
| `transit_native_promotion_c33_64_batch_completed_history` | 167 | 0.018153935 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | 50 | 0.347222222 |

| Quantity | Value |
|---|---:|
| `delta` | 0.001497772 |
| `mapped_capacity_usable_for_theorem` | true |
| `global_coverage_usable_for_theorem` | false |
| `usable_for_global_theorem` | false |

## Unmapped Reasons

| Reason | Count |
|---|---:|
| `scheduleurmbench_unknown` | 155 |
| `unmapped_cpu` | 1776 |
| `unmapped_gpu` | 19 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
