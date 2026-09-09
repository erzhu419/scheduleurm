# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|full_loaded|same_workload_to_capacity_boundary` | `jtl311linux` | `hybrid_rl_bapr_ant` | `bapr_ant` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[]` | `p1=0.242136, p2=0.231653, p3=0, p4=0.230071, p5=0.319606, p6=0.231403` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
