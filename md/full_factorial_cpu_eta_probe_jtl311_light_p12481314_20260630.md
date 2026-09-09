# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q00_low_cpu_low_gpu|light_control_local|light|empty|none` | `jtl311linux` | `light_control_local` | `light` | `empty` | `[1, 2, 4, 8, 13, 14]` | true | `p1=2374.78, p2=4734.39, p4=9491.21, p8=14250, p13=15357.5, p14=15165.5` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
