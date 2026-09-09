# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|empty|none` | `jtl110gpu` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `empty` | `[1, 2, 3, 4, 5, 6, 8]` | true | `[]` | `p1=1856.17, p2=3751.43, p3=5102.46, p4=5812.96, p5=6396.61, p6=7215.26, p8=11403.2` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
