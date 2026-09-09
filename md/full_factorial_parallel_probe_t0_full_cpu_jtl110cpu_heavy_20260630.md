# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|full_loaded|same_workload_to_capacity_boundary` | `jtl110cpu` | `cpu_heavy_local_bench` | `cpu` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=30.5515, p2=28.9389, p4=29.1806, p8=27.8069, p16=25.422, p32=21.2419, p64=16.7389, p96=10.6913, p128=10.4873, p180=7.43773, p192=7.15032` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
