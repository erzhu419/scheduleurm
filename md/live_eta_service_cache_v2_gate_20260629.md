# Live ETA Service Cache v2 Gate

- Status: `LIVE_ETA_SERVICE_CACHE_V2_READY`
- Pass: `true`
- Admitted rows: `10`
- Capacity/unstable boundary rows: `2`
- Missing rows: `0`

| Node bucket | Profile | Valid | Ready/running | Aggregate stable rate | Boundary reason |
|---|---:|---:|---:|---:|---|
| `jtl110gpu:gpu_3080ti_12gb_dual` | 1 | true | 2/2 | 49.3363 | `` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | 2 | true | 4/4 | 50.0441 | `` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | 4 | false | 0/8 | 0 | `capacity_or_unstable_profile` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | 1 | true | 2/2 | 26.5013 | `` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | 2 | true | 4/4 | 23.6441 | `` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | 4 | false | 4/8 | 22.0062 | `capacity_or_unstable_profile` |
| `node007-direct:gpu_node007_4x12gb` | 1 | true | 4/4 | 126.909 | `` |
| `node007-direct:gpu_node007_4x12gb` | 2 | true | 8/8 | 115.983 | `` |
| `node007-direct:gpu_node007_4x12gb` | 4 | true | 16/16 | 114.339 | `` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | 1 | true | 2/2 | 575.089 | `` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | 2 | true | 4/4 | 1264.17 | `` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | 1 | true | 2/2 | 440.197 | `` |

Task-native tqdm/ScheduleurmStableRate CNN and LLM rows for jtl110gpu, jtl311linux, and node007-direct where reachable. Invalid p3/p4 rows are retained as capacity/unstable boundaries and are not theorem-facing service rows.
