# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `3`
- Launched rows: `3`
- Admitted rows: `2`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|full_loaded|same_workload_to_capacity_boundary` | `jtl311linux` | `gpu_cnn_torch_resnet50` | `cnn` | `full_loaded` | `[1, 2, 3, 4, 5, 6, 8]` | false | `[4]` | `p1=26.6466, p2=31.3603, p3=30.0457, p4=34.2072, p5=30.3902, p6=23.7285, p8=23.4138` |
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|full_loaded|same_workload_to_capacity_boundary` | `jtl311linux` | `gpu_llm_distilgpt2` | `llm` | `full_loaded` | `[1, 2, 3, 4]` | true | `[]` | `p1=574.785, p2=909.55, p3=1120.31, p4=1234.86` |
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|full_loaded|same_workload_to_capacity_boundary` | `jtl311linux` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `full_loaded` | `[1, 2, 3, 4, 5, 6, 8]` | true | `[]` | `p1=782.988, p2=857.55, p3=1411.82, p4=1935.84, p5=2659.66, p6=3889.71, p8=4472.53` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
