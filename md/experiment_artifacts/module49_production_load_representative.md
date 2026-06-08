# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = True
record_count_window = 5176
mapped_task_count = 2552
representative_mapped_task_count = 1588
unmapped_task_count = 2624
mapped_fraction = 0.49304482225656876
```

| Workload | Count | Lambda |
|---|---:|---:|
| `cpu_heavy_local_bench` | 118 | 0.045524691 |
| `freqduet_cpu_ablation_c17_32` | 156 | 0.004333333 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 2001 | 0.061759259 |
| `light_control_local` | 206 | 0.794753086 |

| Quantity | Value |
|---|---:|
| `delta` | 0.272849952 |
| `mapped_capacity_usable_for_theorem` | true |
| `global_coverage_usable_for_theorem` | false |
| `usable_for_global_theorem` | false |

## Unmapped Reasons

| Reason | Count |
|---|---:|
| `scheduleurmbench_unknown` | 139 |
| `unmapped_cpu` | 2466 |
| `unmapped_gpu` | 19 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
