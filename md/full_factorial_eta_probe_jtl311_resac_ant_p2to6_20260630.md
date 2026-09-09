# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q11_high_cpu_high_gpu|hybrid_rl_resac_ant|ant|empty|none` | `jtl311linux` | `hybrid_rl_resac_ant` | `ant` | `empty` | `[2, 3, 4, 5, 6]` | true | `[]` | `p2=0.365466, p3=0.36515, p4=0.365293, p5=0.368729, p6=0.369236` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
