# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `3`
- Launched rows: `3`
- Admitted rows: `0`
- Boundary rows: `3`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|half_loaded|same_workload_half_capacity` | `node007-direct` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[2, 3, 4, 8]` | false | `[8]` | `p2=183.935, p3=204.916, p4=197.662, p8=158.085` |
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|half_loaded|same_workload_half_capacity` | `node007-direct` | `gpu_llm_distilgpt2` | `llm` | `half_loaded` | `[2, 3, 4]` | false | `[3]` | `p2=2730.07, p3=3706.25, p4=4186.85` |
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|half_loaded|same_workload_half_capacity` | `node007-direct` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `half_loaded` | `[2, 3, 4, 8]` | false | `[8]` | `p2=2580.31, p3=3252.01, p4=3835.08, p8=6925.22` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
