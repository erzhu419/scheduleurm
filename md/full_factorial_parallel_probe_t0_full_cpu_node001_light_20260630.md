# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q00_low_cpu_low_gpu|light_control_local|light|full_loaded|same_workload_to_capacity_boundary` | `node001` | `light_control_local` | `light` | `full_loaded` | `[1, 2, 4, 8, 13, 14]` | true | `p1=1533.68, p2=3014.68, p4=5955.08, p8=11785.5, p13=18401.5, p14=19804.6` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
