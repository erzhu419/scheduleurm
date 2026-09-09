# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_halfcheetah|halfcheetah|empty|none` | `jtl110gpu` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `empty` | `[3, 4, 5, 6]` | true | `[]` | `p3=0.283857, p4=0.283862, p5=0.283819, p6=0.283461` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
