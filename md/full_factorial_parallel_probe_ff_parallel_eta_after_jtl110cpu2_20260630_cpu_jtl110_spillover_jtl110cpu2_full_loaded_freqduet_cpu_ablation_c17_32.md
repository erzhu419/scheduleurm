# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_jtl110_128c|q10_high_cpu_low_gpu|freqduet_cpu_ablation_c17_32|freqduet|full_loaded|same_workload_to_capacity_boundary` | `jtl110cpu2` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=21.7267, p2=20.7218, p4=19.5488, p8=19.1465, p16=17.0801, p32=16.3922, p64=15.0675, p96=10.8333, p128=10.6411, p180=6.67625, p192=5.00331` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
