# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `hpc_equiv_fill|node003|cpu_heavy_local_bench|full_loaded` | `node003` | `cpu_heavy_local_bench` | `cpu` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=29.1064, p2=24.6564, p4=18.3377, p8=18.3527, p16=18.4934, p32=17.2547, p64=13.757, p96=9.72997, p128=8.73375, p180=7.47734, p192=7.16011` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
