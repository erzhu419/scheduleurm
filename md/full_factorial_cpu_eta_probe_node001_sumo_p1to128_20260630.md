# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|sumo_eval_cpu|sumo|empty|none` | `node001` | `sumo_eval_cpu` | `sumo` | `empty` | `[1, 2, 4, 8, 16, 32, 64, 96, 128]` | false | `p1=36.9516, p2=0, p4=0, p8=21.583, p16=20.8402, p32=20.5245, p64=17.7507, p96=11.2178, p128=10.4896` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
