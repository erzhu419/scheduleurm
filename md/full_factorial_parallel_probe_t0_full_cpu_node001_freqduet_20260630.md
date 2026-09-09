# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|freqduet_cpu_ablation_c17_32|freqduet|full_loaded|same_workload_to_capacity_boundary` | `node001` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=19.92, p2=16.9927, p4=14.368, p8=14.414, p16=14.5668, p32=13.3182, p64=8.9157, p96=7.17599, p128=5.91886, p180=4.44371, p192=3.73341` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
