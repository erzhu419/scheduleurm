# Gavel Native Performance Microbaseline

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `usable_native_performance_sample_count` | 5 |
| `native_gavel_simulator_microbaseline_ready` | false |
| `same_workload_trace_schema_ready` | true |
| `service_unit_equivalence_ready` | false |
| `direct_full_stack_performance_ready` | false |

## Native Rows

| Taskset | Policy | Jobs | Status | Completed | Avg JCT s | Makespan s | Utilization | Usable |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `q01_gpu_bound_cnn_resnet50` | `fifo` | 64 | `TIMEOUT_PARTIAL_PROGRESS` | 2 | 0 | 0 | 0 | false |
| `q01_gpu_bound_cnn_resnet50` | `isolated` | 64 | `PASS` | 64 | 187.593 | 369.415 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | `max_min_fairness` | 64 | `PASS` | 64 | 187.593 | 369.415 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | `finish_time_fairness` | 64 | `PASS` | 64 | 187.593 | 369.415 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | `min_total_duration` | 64 | `PASS` | 64 | 187.593 | 369.415 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | `max_sum_throughput_perf` | 64 | `PASS` | 64 | 187.593 | 369.415 | 1 | true |

## Blockers

- q01_gpu_bound_cnn_resnet50 policy fifo is not a usable native performance row: TIMEOUT_PARTIAL_PROGRESS
- Gavel native simulator performance rows are not yet Scheduleurm measured-service-unit equivalence.
- This gate does not run Pollux, Sia, IADeep, or Salus full production stacks.

## Scope

Runs Gavel's native trace simulator on bounded Scheduleurm-exported trace windows in an isolated copy.  A pass gives native simulator completion/JCT evidence for those finite windows only.  It does not validate Scheduleurm measured-service units against Gavel's throughput tables, and it is not a direct full-stack production-system comparison.
