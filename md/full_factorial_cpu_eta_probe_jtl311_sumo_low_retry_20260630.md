# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q10_high_cpu_low_gpu|sumo_eval_cpu|sumo|empty|none` | `jtl311linux` | `sumo_eval_cpu` | `sumo` | `empty` | `[1, 2, 4, 8]` | false | `p1=74.0729, p2=0, p4=0, p8=31.763` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
