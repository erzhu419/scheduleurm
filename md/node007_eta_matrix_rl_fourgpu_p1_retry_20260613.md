# Node007 ETA Matrix

Run: `node007_eta_matrix_rl_fourgpu_p1_retry_20260613`

ETA source: task-native tqdm/progress logs parsed by `progress_wrapper`; `tui-top` ETA is not used.

| Workload | Case | GPUs | Profile | Stable tasks | Aggregate rate | Mean rate | Seconds/unit range | Valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rl_resac_ant` | `four_gpu_single_task_each` | 4 | 1 | 4/4 | 1.16807 | 0.292017 | 3.4-3.5 s/iter | true |

Overall pass: `true`
