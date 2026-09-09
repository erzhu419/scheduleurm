# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `3`
- Launched rows: `3`
- Admitted rows: `2`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|half_loaded|same_workload_half_capacity` | `jtl311linux` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[3, 4, 8]` | false | `[4]` | `p3=37.7073, p4=40.9053, p8=0` |
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|half_loaded|same_workload_half_capacity` | `jtl311linux` | `gpu_llm_distilgpt2` | `llm` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=902.708, p3=1014.26, p4=1123.03` |
| `gpu_rtx2080_8gb_dual_cpu_fast|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|half_loaded|same_workload_half_capacity` | `jtl311linux` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `half_loaded` | `[2, 3, 4, 8]` | true | `[]` | `p2=872.52, p3=836.222, p4=1031.95, p8=2505.08` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
