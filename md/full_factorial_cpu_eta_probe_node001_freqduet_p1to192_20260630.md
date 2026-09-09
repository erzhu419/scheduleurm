# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|freqduet_cpu_ablation_c17_32|freqduet|empty|none` | `node001` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `empty` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | false | `p1=20.3363, p2=0, p4=14.1624, p8=14.1419, p16=14.1376, p32=13.3121, p64=11.7227, p96=7.19033, p128=0, p180=4.5664, p192=0` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
