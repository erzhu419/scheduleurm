# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `3`
- Launched rows: `3`
- Admitted rows: `0`
- Boundary rows: `3`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `gpu_cnn_torch_resnet50` | `cnn` | `full_loaded` | `[1, 2, 3, 4, 5, 6, 8]` | false | `[4]` | `p1=126.684, p2=183.902, p3=219.047, p4=155.108, p5=212.875, p6=153.768, p8=129.241` |
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `gpu_llm_distilgpt2` | `llm` | `full_loaded` | `[1, 2, 3, 4]` | false | `[3]` | `p1=1419.24, p2=2865.32, p3=3593, p4=4333.33` |
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `full_loaded` | `[1, 2, 3, 4, 5, 6, 8]` | false | `[5]` | `p1=1268.44, p2=3050.45, p3=4188.34, p4=4289.32, p5=4840.09, p6=3823.77, p8=3891.92` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
