# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu2` | `gpu_cnn_torch_resnet50` | `cnn` | `full_loaded` | `[5, 6, 8]` | false | `[5]` | `p5=82.1997, p6=90.5003, p8=69.4149` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
