# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `hpc_equiv_fill|node006|sumo_eval_cpu|full_loaded` | `node006` | `sumo_eval_cpu` | `sumo` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128]` | true | `p1=35.1659, p2=33.4775, p4=21.0211, p8=21.2688, p16=20.9699, p32=19.6873, p64=15.3717, p96=12.096, p128=10.155` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
