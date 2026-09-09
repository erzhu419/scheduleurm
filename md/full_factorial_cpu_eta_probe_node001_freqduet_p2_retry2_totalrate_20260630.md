# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `cpu_hpc_192c|q10_high_cpu_low_gpu|freqduet_cpu_ablation_c17_32|freqduet|empty|none` | `node001` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `empty` | `[2]` | true | `p2=17.5393` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
