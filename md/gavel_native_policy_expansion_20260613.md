# Gavel Native Performance Microbaseline

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `usable_native_performance_sample_count` | 2 |
| `native_gavel_simulator_microbaseline_ready` | false |
| `same_workload_trace_schema_ready` | true |
| `service_unit_equivalence_ready` | false |
| `direct_full_stack_performance_ready` | false |

## Native Rows

| Taskset | Policy | Jobs | Status | Completed | Avg JCT s | Makespan s | Utilization | Usable |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | `fifo` | 4 | `PASS` | 4 | 25.464 | 25.464 | 1 | true |
| `q01_gpu_bound_compute` | `max_min_fairness` | 4 | `FAIL` | 0 | 0 | 0 | 0 | false |
| `q01_gpu_bound_compute` | `finish_time_fairness` | 4 | `FAIL` | 0 | 0 | 0 | 0 | false |
| `q01_gpu_bound_compute` | `min_total_duration` | 4 | `FAIL` | 0 | 0 | 0 | 0 | false |
| `q11_cpu_gpu_coupled` | `fifo` | 8 | `PASS` | 8 | 11.149 | 11.149 | 1 | true |
| `q11_cpu_gpu_coupled` | `max_min_fairness` | 8 | `FAIL` | 0 | 0 | 0 | 0 | false |
| `q11_cpu_gpu_coupled` | `finish_time_fairness` | 8 | `FAIL` | 0 | 0 | 0 | 0 | false |
| `q11_cpu_gpu_coupled` | `min_total_duration` | 8 | `FAIL` | 0 | 0 | 0 | 0 | false |

## Blockers

- q01_gpu_bound_compute policy max_min_fairness is not a usable native performance row: FAIL
- q01_gpu_bound_compute policy finish_time_fairness is not a usable native performance row: FAIL
- q01_gpu_bound_compute policy min_total_duration is not a usable native performance row: FAIL
- q11_cpu_gpu_coupled policy max_min_fairness is not a usable native performance row: FAIL
- q11_cpu_gpu_coupled policy finish_time_fairness is not a usable native performance row: FAIL
- q11_cpu_gpu_coupled policy min_total_duration is not a usable native performance row: FAIL
- Gavel native simulator performance rows are not yet Scheduleurm measured-service-unit equivalence.
- This gate does not run Pollux, Sia, IADeep, or Salus full production stacks.

## Scope

Runs Gavel's native trace simulator on bounded Scheduleurm-exported trace windows in an isolated copy.  A pass gives native simulator completion/JCT evidence for those finite windows only.  It does not validate Scheduleurm measured-service units against Gavel's throughput tables, and it is not a direct full-stack production-system comparison.
