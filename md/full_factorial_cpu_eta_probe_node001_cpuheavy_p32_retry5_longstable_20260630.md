# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|empty|none` | `node001` | `cpu_heavy_local_bench` | `cpu` | `empty` | `[32]` | false | `p32=0` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
