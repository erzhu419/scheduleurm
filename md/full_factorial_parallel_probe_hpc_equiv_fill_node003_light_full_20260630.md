# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `hpc_equiv_fill|node003|light_control_local|full_loaded` | `node003` | `light_control_local` | `light` | `full_loaded` | `[1, 2, 4, 8, 13, 14]` | true | `p1=1410.62, p2=3069.02, p4=6075.05, p8=12050.8, p13=18785.8, p14=20277.8` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
