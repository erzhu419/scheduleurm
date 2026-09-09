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
| `q01_gpu_bound_cnn_resnet50` | 1 | 1.02273 | 1.02273 | 1 | 0.974136 | 0.974136 | true |
| `q01_gpu_bound_llm_inference` | 2.782 | 2.80536 | 1.0084 | 2.64157 | 2.66256 | 1.00795 | true |
| `q01_gpu_model_portfolio` | 1.61087 | 1.61087 | 1 | 1.58997 | 1.59088 | 1.00057 | true |
| `node007_task_native_cnn` | 1.10727 | 1.22211 | 1.10371 | 1.08048 | 1.12054 | 1.03708 | true |
| `node007_task_native_llm` | 1.88403 | 1.97425 | 1.04788 | 1.83691 | 1.84791 | 1.00599 | true |
| `node007_task_native_eta_portfolio` | 1.33304 | 1.54067 | 1.15576 | 1.30907 | 1.3005 | 0.993458 | true |
| `q10_cpu_host_bound` | 1.53114 | 1.53114 | 1 | 1.53517 | 1.53517 | 1 | true |
| `q11_cpu_gpu_coupled` | 1 | 1.00966 | 1.00966 | 1 | 0.998515 | 0.998515 | true |
| `hybrid_research_portfolio` | 1 | 1.01237 | 1.01237 | 1.16077 | 1.16206 | 1.00111 | true |

## Scope

The gate covers measured Scheduleurm replay tasksets and synthetic global-dispatch/state-cache/LCB certificates. It is not a production launch claim and not a direct Gavel/Pollux/Sia/IADeep binary result.
