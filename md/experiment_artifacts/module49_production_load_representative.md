# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = True
record_count_window = 5527
mapped_task_count = 3523
representative_mapped_task_count = 1812
unmapped_task_count = 2004
mapped_fraction = 0.6374163198842048
```

| Workload | Count | Lambda |
|---|---:|---:|
| `bamor_cpu_training_c3_8_completed_history` | 163 | 9.958333333 |
| `cpu_heavy_local_bench` | 342 | 0.131944444 |
| `freqduet_cpu_ablation_c17_32` | 173 | 0.004805556 |
| `freqduet_cpu_ablation_c33_64_completed_history` | 63 | 0.060239198 |
| `freqduet_cpu_ablation_c3_8_completed_history` | 42 | 0.006945602 |
| `freqduet_cpu_ablation_c9_16` | 129 | 0.068325617 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |
| `freqduet_runner_v3_c_le2_completed_history` | 84 | 0.001798225 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 2001 | 0.061759259 |
| `light_control_local` | 206 | 0.794753086 |
| `sumo_eval_simple_sac_c_le2` | 71 | 0.000027392 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | 128 | 0.014145833 |
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
| `unmapped_cpu` | 1830 |
| `unmapped_gpu` | 19 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
