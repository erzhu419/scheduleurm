# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|empty|none` | `node001` | `cpu_heavy_local_bench` | `cpu` | `empty` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | false | `p1=29.3496, p2=0, p4=18.3013, p8=18.3952, p16=18.0283, p32=0, p64=0, p96=9.55008, p128=8.40236, p180=0, p192=5.81746` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
