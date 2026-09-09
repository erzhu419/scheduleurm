# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q10_high_cpu_low_gpu|sumo_eval_cpu|sumo|full_loaded|same_workload_to_capacity_boundary` | `jtl110cpu` | `sumo_eval_cpu` | `sumo` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128]` | true | `p1=40.6747, p2=37.1622, p4=37.024, p8=34.7595, p16=31.3645, p32=26.7495, p64=19.9385, p96=14.1456, p128=12.0184` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
