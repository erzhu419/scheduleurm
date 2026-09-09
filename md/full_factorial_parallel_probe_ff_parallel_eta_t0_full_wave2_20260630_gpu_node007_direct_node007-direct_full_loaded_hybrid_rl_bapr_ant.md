# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `hybrid_rl_bapr_ant` | `bapr_ant` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[5]` | `p1=0.569378, p2=0.627578, p3=0.725732, p4=0.75667, p5=0.752775, p6=0.811204` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
