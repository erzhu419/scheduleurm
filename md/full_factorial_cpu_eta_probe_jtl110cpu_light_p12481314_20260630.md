# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q00_low_cpu_low_gpu|light_control_local|light|empty|none` | `jtl110cpu` | `light_control_local` | `light` | `empty` | `[1, 2, 4, 8, 13, 14]` | true | `p1=884.366, p2=1581.66, p4=2484.34, p8=4316.84, p13=6320.96, p14=6533.18` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
