# Node007 ETA Matrix

Run: `node007_eta_matrix_llm_20260613`

ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.

| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `llm_torch_transformer` | `single_gpu_single_task` | 1 | 1 | 1/1 | 411.641 | 411.641 | 0.002429-0.002429 s/step | true |
| `llm_torch_transformer` | `single_gpu_multi_task` | 1 | 2 | 2/2 | 643.968 | 321.984 | 0.002424-0.00432 s/step | true |
| `llm_torch_transformer` | `single_gpu_multi_task` | 1 | 3 | 3/3 | 1054.49 | 351.497 | 0.002503-0.003445 s/step | true |
| `llm_torch_transformer` | `single_gpu_multi_task` | 1 | 4 | 4/4 | 1217.4 | 304.35 | 0.002618-0.005152 s/step | true |
| `llm_torch_transformer` | `four_gpu_single_task_each` | 4 | 1 | 4/4 | 1238.22 | 309.554 | 0.002794-0.003701 s/step | true |
| `llm_torch_transformer` | `four_gpu_two_tasks_each` | 4 | 2 | 8/8 | 2518.29 | 314.786 | 0.002549-0.003958 s/step | true |

Overall pass: `true`
