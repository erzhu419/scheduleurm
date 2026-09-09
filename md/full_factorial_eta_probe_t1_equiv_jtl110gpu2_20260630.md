# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual_equiv|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|empty|none` | `jtl110gpu2` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[4, 5, 6, 8]` | false | `[5]` | `p4=128.654, p5=95.0323, p6=140.717, p8=149.052` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
