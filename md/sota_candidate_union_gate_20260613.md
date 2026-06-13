# SOTA Candidate-Union Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `SOTA_CANDIDATE_UNION_PASS` |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `scenario_count` | 16 |
| `union_makespan_beats_sota_envelope_all` | true |
| `union_mean_flow_beats_sota_envelope_all` | true |
| `guarded_union_not_pareto_dominated_all` | true |
| `single_guarded_union_beats_both_envelopes_all` | false |
| `adaptive_scalarized_union_not_pareto_dominated_all` | true |
| `adaptive_scalarized_union_beats_both_envelopes_all` | false |

## Scenario Envelope

| Taskset | Arrival | Best SOTA makespan | Union makespan | Ratio | Best SOTA flow | Union flow | Ratio | Guarded union Pareto dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | `static` | 8517.37 | 8517.37 | 1 | 4256.93 | 4256.93 | 1 | false |
| `q00_light_control` | `poisson` | 8533.21 | 8533.21 | 1 | 424.267 | 424.267 | 1 | false |
| `q01_gpu_bound_compute` | `static` | 1634.02 | 1634.02 | 1 | 877.959 | 877.959 | 1 | false |
| `q01_gpu_bound_compute` | `poisson` | 2576.55 | 2580.72 | 0.998383 | 100.688 | 100.688 | 1 | false |
| `q01_gpu_bound_cnn_resnet50` | `static` | 50.7545 | 50.7545 | 1 | 27.44 | 27.4544 | 0.999476 | false |
| `q01_gpu_bound_cnn_resnet50` | `poisson` | 3326.88 | 3326.88 | 1 | 2.72852 | 2.72852 | 1 | false |
| `q01_gpu_bound_llm_inference` | `static` | 2.06026 | 2.06026 | 1 | 1.10738 | 1.10738 | 1 | false |
| `q01_gpu_bound_llm_inference` | `poisson` | 2339.93 | 2339.93 | 1 | 0.210106 | 0.210106 | 1 | false |
| `q01_gpu_model_portfolio` | `static` | 1095.83 | 1095.83 | 1 | 115.008 | 115.01 | 0.999988 | false |
| `q01_gpu_model_portfolio` | `poisson` | 2682.16 | 2682.16 | 1 | 19.4149 | 19.4149 | 1 | false |
| `q10_cpu_host_bound` | `static` | 4754.53 | 4754.53 | 1 | 2400.61 | 2400.61 | 1 | false |
| `q10_cpu_host_bound` | `poisson` | 4807.22 | 4807.22 | 1 | 569.804 | 569.804 | 1 | false |
| `q11_cpu_gpu_coupled` | `static` | 38336.4 | 38382.4 | 0.998801 | 19394.4 | 19394.4 | 1 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 38135.9 | 38184.5 | 0.998727 | 18145.7 | 18145.7 | 1 | false |
| `hybrid_research_portfolio` | `static` | 28514.5 | 28514.5 | 1 | 4825.15 | 4825.15 | 1 | false |
| `hybrid_research_portfolio` | `poisson` | 28555.6 | 28555.6 | 1 | 3542.9 | 3542.9 | 1 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `calibrated_backlog_aware_guarded` | `candidate` |  | 174674 | 3576.04 | 1 | 1 |
| `legacy_fixed_caps` | `legacy` |  | 388938 | 17038.5 | 2.22665 | 4.76464 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 174302 | 3575.57 | 0.997868 | 0.999869 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 174265 | 3578.3 | 0.99766 | 1.00063 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173962 | 3585.77 | 0.99592 | 1.00272 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 174302 | 3575.57 | 0.997868 | 0.999869 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis | 174225 | 3581.4 | 0.997426 | 1.0015 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia | 174987 | 3597.14 | 1.00179 | 1.0059 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 174150 | 3575.89 | 0.997001 | 0.999958 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins | 174827 | 3581.82 | 1.00087 | 1.00162 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva | 173962 | 3585.77 | 0.99592 | 1.00272 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux | 174302 | 3575.57 | 0.997868 | 0.999869 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 174674 | 3576.04 | 1 | 1 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family action-union dominance/envelope rows, not direct full-stack external binary superiority.
