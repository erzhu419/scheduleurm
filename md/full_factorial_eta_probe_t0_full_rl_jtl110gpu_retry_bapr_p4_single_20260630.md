# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_bapr_ant` | `bapr_ant` | `full_loaded` | `[4]` | true | `[]` | `p4=0.663438` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
