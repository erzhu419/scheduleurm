# Node007 ETA Matrix

Run: `node007_eta_matrix_rl_20260613`

ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.

| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rl_resac_ant` | `single_gpu_single_task` | 1 | 1 | 1/1 | 0.294118 | 0.294118 | 3.4-3.4 s/iter | true |
| `rl_resac_ant` | `single_gpu_multi_task` | 1 | 2 | 2/2 | 0.290404 | 0.145202 | 6.6-7.2 s/iter | true |
| `rl_resac_ant` | `single_gpu_multi_task` | 1 | 3 | 3/3 | 0.378233 | 0.126078 | 6.7-11.1 s/iter | true |
| `rl_resac_ant` | `single_gpu_multi_task` | 1 | 4 | 4/4 | 0.353906 | 0.0884765 | 9.9-14.8 s/iter | true |
| `rl_resac_ant` | `single_gpu_multi_task` | 1 | 5 | 5/5 | 0.490993 | 0.0981986 | 6.9-18.5 s/iter | true |
| `rl_resac_ant` | `four_gpu_single_task_each` | 4 | 1 | 0/4 | 0 | 0 | 0-0 s/iter | false |
| `rl_resac_ant` | `four_gpu_two_tasks_each` | 4 | 2 | 8/8 | 1.57555 | 0.196944 | 3.4-7.4 s/iter | true |

Overall pass: `false`
