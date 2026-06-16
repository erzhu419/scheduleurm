# Gavel Native Performance Microbaseline

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `usable_native_performance_sample_count` | 2 |
| `native_gavel_simulator_microbaseline_ready` | true |
| `same_workload_trace_schema_ready` | true |
| `service_unit_equivalence_ready` | false |
| `direct_full_stack_performance_ready` | false |

## Native Rows

| Taskset | Policy | Jobs | Status | Completed | Avg JCT s | Makespan s | Utilization | Usable |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | `fifo` | 4 | `PASS` | 4 | 25.464 | 25.464 | 1 | true |
| `q11_cpu_gpu_coupled` | `fifo` | 8 | `PASS` | 8 | 11.149 | 11.149 | 1 | true |

## Blockers

- Gavel native simulator performance rows are not yet Scheduleurm measured-service-unit equivalence.
- This gate does not run Pollux, Sia, IADeep, or Salus full production stacks.

## Scope

Runs Gavel's native trace simulator on bounded Scheduleurm-exported trace windows in an isolated copy.  A pass gives native simulator completion/JCT evidence for those finite windows only.  It does not validate Scheduleurm measured-service units against Gavel's throughput tables, and it is not a direct full-stack production-system comparison.
