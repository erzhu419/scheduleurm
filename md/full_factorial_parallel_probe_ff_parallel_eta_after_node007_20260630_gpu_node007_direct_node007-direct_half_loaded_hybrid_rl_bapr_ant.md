# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|half_loaded|same_workload_half_capacity` | `node007-direct` | `hybrid_rl_bapr_ant` | `bapr_ant` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=0.635237, p3=0.721711, p4=0.754978` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
