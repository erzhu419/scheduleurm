# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q10_high_cpu_low_gpu|freqduet_cpu_ablation_c17_32|freqduet|empty|none` | `jtl311linux` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `empty` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | false | `p1=42.7502, p2=43.5029, p4=0, p8=18.9146, p16=12.2613, p32=6.96862, p64=3.54625, p96=2.39986, p128=1.79881, p180=1.34812, p192=1.2883` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
