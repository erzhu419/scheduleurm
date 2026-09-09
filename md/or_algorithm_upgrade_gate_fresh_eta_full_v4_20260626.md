# OR Algorithm Upgrade Gate

This gate validates the optional algorithm-layer upgrade. It does not change the legacy scheduler default.

| Quantity | Value |
|---|---:|
| `pass` | false |
| `status` | `OR_ALGORITHM_UPGRADE_REVIEW` |
| `regression_count` | 0 |
| `improvement_count` | 5 |
| `global_batch_scheduler_hook_ready` | true |
| `state_dependent_cache_ready` | true |
| `online_lcb_eta_ready` | true |

## Replay Rows

| Taskset | Current vs legacy makespan | Upgraded vs legacy makespan | Upgraded/current makespan | Current vs legacy flow | Upgraded vs legacy flow | Upgraded/current flow | SOTA Pareto safe (tol.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | 1.46593 | 1.46593 | 1 | 1.54221 | 1.54221 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | 1.13576 | 1.24433 | 1.0956 | 1.12589 | 1.1445 | 1.01653 | true |
| `q01_gpu_bound_llm_inference` | 2.68778 | 2.68778 | 1 | 2.64157 | 2.64157 | 1 | false |
| `q01_gpu_model_portfolio` | 1.61087 | 1.61087 | 1 | 1.60175 | 1.60313 | 1.00086 | true |
| `node007_task_native_cnn` | 1.10727 | 1.28261 | 1.15835 | 1.08048 | 1.08498 | 1.00416 | true |
| `node007_task_native_llm` | 1.88403 | 1.88403 | 1 | 1.83691 | 1.83691 | 1 | false |
| `node007_task_native_eta_portfolio` | 1.33304 | 1.54067 | 1.15576 | 1.30907 | 1.30049 | 0.993452 | true |
| `q10_cpu_host_bound` | 1.53114 | 1.53114 | 1 | 1.53517 | 1.53517 | 1 | true |
| `q11_cpu_gpu_coupled` | 1 | 1.00954 | 1.00954 | 1 | 1.04677 | 1.04677 | true |
| `hybrid_research_portfolio` | 1 | 1.01245 | 1.01245 | 1.1608 | 1.20698 | 1.03978 | true |

## Scope

The gate covers measured Scheduleurm replay tasksets and synthetic global-dispatch/state-cache/LCB certificates. It is not a production launch claim and not a direct Gavel/Pollux/Sia/IADeep binary result.
