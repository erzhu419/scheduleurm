# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|full_loaded|same_workload_to_capacity_boundary` | `node001` | `cpu_heavy_local_bench` | `cpu` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=28.7606, p2=25.1403, p4=18.1479, p8=18.3894, p16=18.2772, p32=18.3522, p64=11.9139, p96=9.52029, p128=7.78731, p180=5.31256, p192=5.05789` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
