# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q10_high_cpu_low_gpu|cpu_heavy_local_bench|cpu|empty|none` | `jtl311linux` | `cpu_heavy_local_bench` | `cpu` | `empty` | `[16, 32, 64, 96, 128, 180, 192]` | true | `p16=17.5817, p32=9.87782, p64=5.089, p96=3.4046, p128=2.5553, p180=1.81288, p192=1.69769` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
