# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q00_low_cpu_low_gpu|light_control_local|light|full_loaded|same_workload_to_capacity_boundary` | `jtl110cpu` | `light_control_local` | `light` | `full_loaded` | `[1, 2, 4, 8, 13, 14]` | true | `p1=938.101, p2=1351.92, p4=2961.17, p8=6107.48, p13=9245.51, p14=10253.9` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
