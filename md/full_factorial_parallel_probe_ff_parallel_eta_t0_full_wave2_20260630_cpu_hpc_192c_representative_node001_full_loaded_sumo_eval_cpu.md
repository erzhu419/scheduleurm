# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|sumo_eval_cpu|sumo|full_loaded|same_workload_to_capacity_boundary` | `node001` | `sumo_eval_cpu` | `sumo` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128]` | true | `p1=35.9424, p2=30.4646, p4=21.4609, p8=22.1494, p16=21.9644, p32=19.6278, p64=12.7508, p96=10.7831, p128=10.1992` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
