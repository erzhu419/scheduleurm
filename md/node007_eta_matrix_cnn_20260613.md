# Node007 ETA Matrix

Run: `node007_eta_matrix_cnn_20260613`

ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.

| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cnn_torch_gpu` | `single_gpu_single_task` | 1 | 1 | 0/0 | 0 | 0 | 0-0 s/None | false |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 1 | 2,3,4 | 0/0 | 0 | 0 | 0-0 s/None | false |
| `cnn_torch_gpu` | `four_gpu_single_task_each` | 4 | 1 | 0/0 | 0 | 0 | 0-0 s/None | false |
| `cnn_torch_gpu` | `four_gpu_two_tasks_each` | 4 | 2 | 0/0 | 0 | 0 | 0-0 s/None | false |

Overall pass: `false`
