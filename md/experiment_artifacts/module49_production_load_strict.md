# Production Load Capacity Certificate

```text
window_days = 30.0
include_representative = False
record_count_window = 5159
mapped_task_count = 807
representative_mapped_task_count = 0
unmapped_task_count = 4352
mapped_fraction = 0.15642566388835047
```

| Workload | Count | Lambda |
|---|---:|---:|
| `cpu_heavy_local_bench` | 55 | 0.021219136 |
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
| `scheduleurmbench_unknown` | 106 |
| `unmapped_cpu` | 2505 |
| `unmapped_gpu` | 1741 |

Positive capacity on mapped measured buckets is a load certificate only for those buckets. Global production stability remains open unless unmapped and representative-mapped tasks are eliminated or separately certified by service measurements.
