# Gavel Physical Same-Workload Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `same_service_scale_ready` | true |
| `scoped_gavel_physical_fullstack_ready` | true |
| `scoped_gavel_physical_same_workload_superiority_ready` | true |
| `direct_fullstack_binary_superiority_ready` | true |
| `native_wall_s` | 2.0166 |
| `gavel_jct_s` | 2.996 |
| `gavel_total_scheduler_wall_s` | 16.0603 |
| `scheduleurm_native_to_gavel_jct_ratio` | 0.673099 |

## Runtime Checks

| Check | Ready | Detail |
|---|---:|---|
| `dependency_imports` | true | `` |
| `extended_imports` | true | `` |
| `protobuf_stubs` | true | `` |
| `preparation` | true | `PREPARED` |
| `native_same_workload` | true | `PASS` |
| `gavel_physical_same_workload` | true | `PASS` |

## Blockers

| Blocker |
|---|
| none |

## Scope

Runs Gavel's physical scheduler/worker/RPC/dispatcher/GavelIterator path on a temporary same-host workload and compares it with the same script run directly.  This is a scoped Gavel physical-runtime row.  It does not certify every Gavel paper workload or a multi-node Gavel cluster.
