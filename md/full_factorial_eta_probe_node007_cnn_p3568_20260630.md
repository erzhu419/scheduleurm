# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Rates |
|---|---|---|---|---|---:|---:|---|
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|empty|none` | `node007-direct` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[3, 5, 6, 8]` | false | `p3=115.482, p5=114.163, p6=112.951, p8=110.658` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
