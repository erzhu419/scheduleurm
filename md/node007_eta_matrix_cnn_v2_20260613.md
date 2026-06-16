# Node007 ETA Matrix

Run: `node007_eta_matrix_cnn_v2_20260613`

ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.

| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cnn_torch_gpu` | `single_gpu_single_task` | 1 | 1 | 0/0 | 0 | 0 | 0-0 s/None | false |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 1 | 2 | 2/2 | 29.43 | 14.715 | 0.06658-0.0694 s/step | true |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 1 | 3 | 3/3 | 55.59 | 18.53 | 0.03183-0.08489 s/step | true |
| `cnn_torch_gpu` | `single_gpu_multi_task` | 1 | 4 | 4/4 | 29.8 | 7.45 | 0.133-0.1355 s/step | true |
| `cnn_torch_gpu` | `four_gpu_single_task_each` | 4 | 1 | 4/4 | 126.25 | 31.5625 | 0.03147-0.03189 s/step | true |
| `cnn_torch_gpu` | `four_gpu_two_tasks_each` | 4 | 2 | 8/8 | 197.887 | 24.7359 | 0.03143-0.06423 s/step | true |

Overall pass: `false`
