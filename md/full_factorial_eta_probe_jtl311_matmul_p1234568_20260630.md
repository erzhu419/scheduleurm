# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|empty|none` | `jtl311linux` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `empty` | `[1, 2, 3, 4, 5, 6, 8]` | true | `[]` | `p1=783.451, p2=937.157, p3=1143.57, p4=1370.29, p5=1265.72, p6=2085.13, p8=2731.71` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
