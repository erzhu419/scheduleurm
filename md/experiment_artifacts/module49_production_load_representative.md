# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = True
record_count_window = 5159
mapped_task_count = 2590
representative_mapped_task_count = 1783
unmapped_task_count = 2569
mapped_fraction = 0.5020352781546812
```

| Workload | Count | Lambda |
|---|---:|---:|
| `cpu_heavy_local_bench` | 118 | 0.045524691 |
| `gpu_heavy_jax_matmul` | 70 | 0.064814815 |
| `hybrid_rl_resac_ant` | 2196 | 0.067777778 |
| `light_control_local` | 206 | 0.794753086 |

| Quantity | Value |
|---|---:|
| `delta` | 0.266831433 |
| `mapped_capacity_usable_for_theorem` | true |
| `global_coverage_usable_for_theorem` | false |
| `usable_for_global_theorem` | false |

## Unmapped Reasons

| Reason | Count |
|---|---:|
| `scheduleurmbench_unknown` | 106 |
| `unmapped_cpu` | 2442 |
| `unmapped_gpu` | 21 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
