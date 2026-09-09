# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|full_loaded|same_workload_to_capacity_boundary` | `jtl110cpu2` | `cpu_heavy_local_bench` | `cpu` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=29.2006, p2=29.6025, p4=27.9289, p8=27.2105, p16=24.5652, p32=23.2391, p64=20.7035, p96=14.3174, p128=15.3791, p180=7.68654, p192=8.95354` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
