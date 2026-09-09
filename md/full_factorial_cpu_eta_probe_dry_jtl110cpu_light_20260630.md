# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_MANIFEST`
- Allow launch: `false`
- Selected rows: `1`
- Launched rows: `0`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q00_low_cpu_low_gpu|light_control_local|light|empty|none` | `jtl110cpu` | `light_control_local` | `light` | `empty` | `[1, 2]` | false | `` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
