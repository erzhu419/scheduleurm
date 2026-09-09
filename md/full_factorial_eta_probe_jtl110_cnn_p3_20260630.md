# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `1`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|empty|none` | `jtl110gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[3]` | true | `p3=50.263` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
