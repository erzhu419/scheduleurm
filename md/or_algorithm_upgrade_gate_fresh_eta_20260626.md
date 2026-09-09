# OR Algorithm Upgrade Gate

This gate validates the optional algorithm-layer upgrade. It does not change the legacy scheduler default.

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `OR_ALGORITHM_UPGRADE_PASS` |
| `regression_count` | 0 |
| `improvement_count` | 7 |
| `global_batch_scheduler_hook_ready` | true |
| `state_dependent_cache_ready` | true |
| `online_lcb_eta_ready` | true |

## Replay Rows

| Taskset | Current vs legacy makespan | Upgraded vs legacy makespan | Upgraded/current makespan | Current vs legacy flow | Upgraded vs legacy flow | Upgraded/current flow | SOTA Pareto safe (tol.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | 1.46593 | 1.46593 | 1 | 1.54221 | 1.54221 | 1 | true |
| `q01_gpu_bound_cnn_resnet50` | 1 | 1.02168 | 1.02168 | 1 | 0.980601 | 0.980601 | true |
| `q01_gpu_bound_llm_inference` | 2.782 | 2.80536 | 1.0084 | 2.64157 | 2.66256 | 1.00795 | true |
| `q01_gpu_model_portfolio` | 1.61087 | 1.61087 | 1 | 1.58997 | 1.59088 | 1.00057 | true |
| `node007_task_native_cnn` | 1.10727 | 1.22211 | 1.10371 | 1.08048 | 1.12054 | 1.03708 | true |
| `node007_task_native_llm` | 1.14511 | 1.15587 | 1.00939 | 1.1442 | 1.13393 | 0.991018 | true |
| `node007_task_native_eta_portfolio` | 1.33304 | 1.54067 | 1.15576 | 1.30756 | 1.299 | 0.993454 | true |
| `q10_cpu_host_bound` | 1.53114 | 1.53114 | 1 | 1.53517 | 1.53517 | 1 | true |
| `q11_cpu_gpu_coupled` | 1 | 1.00966 | 1.00966 | 1 | 0.998515 | 0.998515 | true |
| `hybrid_research_portfolio` | 1 | 1.01237 | 1.01237 | 1.16077 | 1.16206 | 1.00111 | true |

## Scope

The gate covers measured Scheduleurm replay tasksets and synthetic global-dispatch/state-cache/LCB certificates. It is not a production launch claim and not a direct Gavel/Pollux/Sia/IADeep binary result.
