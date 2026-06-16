# Gavel Native Same-Workload Comparison

| Quantity | Value |
|---|---:|
| `pass` | true |
| `same_workload_native_simulator_superiority_ready` | true |
| `direct_fullstack_binary_superiority_ready` | false |
| `eligible_row_count` | 1 |
| `raw_native_simulator_superiority_count` | 1 |

| Taskset | Jobs | Gavel best policy | Gavel makespan | Gavel mean JCT | Scheduleurm policy | Scheduleurm makespan | Scheduleurm mean flow | Raw superior | Eligible |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| `q01_gpu_bound_cnn_resnet50` | 64 | `isolated` | 369.415 | 187.593 | `scheduleurm_sota_union_makespan` | 50.6207 | 27.3043 | true | true |

## Scope

Compares Scheduleurm measured-cache replay to Gavel's native simulator on full-taskset same-workload trace exports.  A ready result supports a scoped native-Gavel-simulator baseline claim.  It is not direct full-stack binary superiority because Gavel's simulator uses its own throughput tables and does not launch the real workload binaries on the Scheduleurm cluster.
