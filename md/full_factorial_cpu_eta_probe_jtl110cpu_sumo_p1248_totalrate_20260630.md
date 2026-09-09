# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q10_high_cpu_low_gpu|sumo_eval_cpu|sumo|empty|none` | `jtl110cpu` | `sumo_eval_cpu` | `sumo` | `empty` | `[1, 2, 4, 8]` | true | `p1=39.9423, p2=38.2302, p4=35.6593, p8=34.3153` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
