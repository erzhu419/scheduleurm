# OR Gate: Module Ablation Suite

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| trace_count | 24 |
| full_candidate_vs_legacy_geomean_makespan | 1.73733 |
| full_candidate_vs_legacy_geomean_mean_flow | 2.99978 |
| ablation_dominator_count | 0 |

## Reviewer Axis Coverage

| Axis | Policy | Role |
|---|---|---|
| legacy rules | `legacy_fixed_caps` | fixed legacy caps and legacy-style profile choices |
| sweetspot hook | `ablation_sweetspot_scalar_hook` | scalar calibrated sweetspot/guarded-knee replay without queue-adaptive MaxWeight |
| support scorer | `ablation_support_only_makespan` | support/makespan objective without delay or queue-adaptive penalty |
| delay tie-break | `ablation_delay_only_mean_flow` | delay/mean-flow objective without support-preserving robust guard |
| statewise guard | `ablation_statewise_interference_guard` | statewise co-location/interference guard without the full adaptive scorer |
| profile penalty | `ablation_robust_no_profile_penalty` | same support-envelope MaxWeight scorer with bounded profile penalty removed |
| tie-break direction | `ablation_robust_high_profile_tiebreak` | same robust scorer and penalty with high-profile tie-break instead of low-profile tie-break |
| full robust lower-service scorer | `calibrated_adaptive_maxweight_penalty` | queue-adaptive robust lower-service MaxWeight before SOTA-action trajectory union |
| SOTA-action union selector | `scheduleurm_sota_union_online_pareto_slack` | full theorem-facing selector over Scheduleurm and SOTA-style trajectory actions |

## Trace Rows

| taskset | trace | jobs | profiles | cand/legacy makespan | cand/legacy flow | dominators |
|---|---|---:|---|---:|---:|---:|
| q00_light_control | q00_light_control_static_seed23 | 512 | `{"light_control_local": 13}` | 11.9137 | 11.7625 | 0 |
| q00_light_control | q00_light_control_poisson_load0.85_seed23 | 512 | `{"light_control_local": 13}` | 9.48638 | 198.632 | 0 |
| q00_light_control | q00_light_control_bursty_load0.85_seed23 | 512 | `{"light_control_local": 13}` | 10.8682 | 50.2395 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_static_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.46619 | 1.5597 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.45993 | 3.49592 | 0 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed23 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.17278 | 2.04608 | 0 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_static_seed23 | 64 | `{"gpu_cnn_torch_resnet50": 5}` | 1.23893 | 1.16149 | 0 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed23 | 64 | `{"gpu_cnn_torch_resnet50": 5}` | 1.22095 | 1.48636 | 0 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed23 | 64 | `{"gpu_cnn_torch_resnet50": 5}` | 1.09506 | 1.20539 | 0 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_static_seed23 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.73224 | 2.62243 | 0 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed23 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.29396 | 8.3159 | 0 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed23 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.13999 | 3.88285 | 0 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_static_seed23 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.61868 | 1.60082 | 0 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed23 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.40066 | 3.05969 | 0 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed23 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.16096 | 2.01759 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_static_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.5311 | 1.53451 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.35388 | 6.80204 | 0 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed23 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.41307 | 2.62219 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_static_seed23 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.0154 | 1.0002 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed23 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01726 | 1.10815 | 0 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed23 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01423 | 1.00181 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_static_seed23 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00286 | 1.15979 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed23 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.02306 | 2.36028 | 0 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed23 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00699 | 1.4861 | 0 |

## Scope

same measured service cache, with each module removed by policy semantics rather than by editing the legacy scheduler
