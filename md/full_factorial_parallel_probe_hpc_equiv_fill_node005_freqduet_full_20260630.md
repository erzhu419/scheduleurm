# Full-Factorial CPU ETA Probe Runner

- Status: `CPU_PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `hpc_equiv_fill|node005|freqduet_cpu_ablation_c17_32|full_loaded` | `node005` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `full_loaded` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | true | `p1=20.0627, p2=17.3894, p4=14.5069, p8=14.408, p16=14.1424, p32=13.4458, p64=11.9629, p96=7.375, p128=6.78677, p180=5.4029, p192=5.20098` |

Controlled full-factorial CPU ETA probes.  Launches bypass the legacy scheduler dispatch path and accept only task-native progress/tqdm rate samples with a stable tail gate.
