# OR Algorithm Upgrade Gate

This gate validates the optional algorithm-layer upgrade. It does not change the legacy scheduler default.

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `OR_ALGORITHM_UPGRADE_PASS` |
| `regression_count` | 0 |
| `improvement_count` | 4 |
| `global_batch_scheduler_hook_ready` | true |
| `state_dependent_cache_ready` | true |
| `online_lcb_eta_ready` | true |

## Replay Rows

| Taskset | Current vs legacy makespan | Upgraded vs legacy makespan | Upgraded/current makespan | Current vs legacy flow | Upgraded vs legacy flow | Upgraded/current flow | SOTA Pareto safe (tol.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | 1.46593 | 1.46593 | 1 | 1.54221 | 1.54221 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | 1 | 1 | 1 | 1 | 1 | 1 | true |
| `q01_gpu_bound_llm_inference` | 2.782 | 2.80536 | 1.0084 | 2.64157 | 2.66256 | 1.00795 | true |
| `q01_gpu_model_portfolio` | 1.61087 | 1.61087 | 1 | 1.58997 | 1.58989 | 0.999951 | true |
| `node007_task_native_cnn` | 1 | 1.05348 | 1.05348 | 1 | 0.996794 | 0.996794 | true |
| `node007_task_native_llm` | 1.13275 | 1.14512 | 1.01092 | 1.11397 | 1.12413 | 1.00912 | true |
| `node007_task_native_eta_portfolio` | 1.17942 | 1.38737 | 1.17632 | 1.1019 | 1.18131 | 1.07207 | true |
| `q10_cpu_host_bound` | 1.53114 | 1.53114 | 1 | 1.53517 | 1.53517 | 1 | true |
| `q11_cpu_gpu_coupled` | 1.38045 | 1.38045 | 1 | 1.40782 | 1.40782 | 1 | true |
| `hybrid_research_portfolio` | 1.38304 | 1.38304 | 1 | 1.42822 | 1.42822 | 1 | true |

## Scope

The gate covers measured Scheduleurm replay tasksets and synthetic global-dispatch/state-cache/LCB certificates. It is not a production launch claim and not a direct Gavel/Pollux/Sia/IADeep binary result.
