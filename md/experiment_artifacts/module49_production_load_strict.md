# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = False
record_count_window = 5196
mapped_task_count = 1093
representative_mapped_task_count = 0
unmapped_task_count = 4103
mapped_fraction = 0.21035411855273287
```

| Workload | Count | Lambda |
|---|---:|---:|
| `cpu_heavy_local_bench` | 55 | 0.021219136 |
| `freqduet_cpu_ablation_c17_32` | 156 | 0.004333333 |
| `freqduet_cpu_ablation_c9_16` | 129 | 0.068325617 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | 1 | 0.000007716 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 476 | 0.014691358 |
| `light_control_local` | 206 | 0.794753086 |

| Quantity | Value |
|---|---:|
| `delta` | 0.319917853 |
| `mapped_capacity_usable_for_theorem` | true |
| `global_coverage_usable_for_theorem` | false |
| `usable_for_global_theorem` | false |

## Unmapped Reasons

| Reason | Count |
|---|---:|
| `scheduleurmbench_unknown` | 155 |
| `unmapped_cpu` | 2404 |
| `unmapped_gpu` | 1544 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
