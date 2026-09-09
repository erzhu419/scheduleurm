# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|empty|none` | `node007-direct` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `empty` | `[1, 2, 3, 4, 5, 6, 8]` | false | `[5]` | `p1=1131.48, p2=2327.36, p3=3189.97, p4=4165.92, p5=4072.31, p6=2244.82, p8=4762.53` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
