# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q00_low_cpu_low_gpu|light_control_local|light|empty|none` | `node001` | `light_control_local` | `light` | `empty` | `[1, 2, 4, 8, 13, 14]` | true | `p1=1372.53, p2=3017.92, p4=6096.92, p8=11575.5, p13=18635.7, p14=19841.6` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
