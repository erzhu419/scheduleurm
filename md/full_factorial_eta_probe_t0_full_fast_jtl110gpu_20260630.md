# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `3`
- Launched rows: `3`
- Admitted rows: `2`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `full_loaded` | `[1, 2, 3, 5, 6, 8]` | false | `[]` | `p1=52.0013, p2=76.8499, p3=92.9722, p5=0, p6=0, p8=0` |
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `gpu_llm_distilgpt2` | `llm` | `full_loaded` | `[1, 2, 3, 4]` | true | `[]` | `p1=625.053, p2=1174.08, p3=1854.29, p4=2228.86` |
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `full_loaded` | `[1, 2, 3, 4, 5, 6, 8]` | true | `[]` | `p1=1867.47, p2=3649.02, p3=4557.89, p4=6902.1, p5=8510.58, p6=9784.13, p8=14216.7` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
