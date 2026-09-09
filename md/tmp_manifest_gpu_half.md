# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_MANIFEST`
- Allow launch: `false`
- Selected rows: `1`
- Launched rows: `0`
- Admitted rows: `0`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[3]` | false | `[]` | `` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
