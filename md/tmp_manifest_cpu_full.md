# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_MANIFEST`
- Allow launch: `false`
- Selected rows: `1`
- Launched rows: `0`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q00_low_cpu_low_gpu|light_control_local|light|full_loaded|same_workload_to_capacity_boundary` | `jtl311linux` | `light_control_local` | `light` | `full_loaded` | `[1]` | false | `` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
