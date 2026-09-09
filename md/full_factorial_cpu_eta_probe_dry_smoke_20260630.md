# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_MANIFEST`
- Allow launch: `false`
- Selected rows: `1`
- Launched rows: `0`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|empty|none` | `jtl311linux` | `cpu_heavy_local_bench` | `cpu` | `empty` | `[1, 2]` | false | `` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
