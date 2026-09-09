# SOTA Candidate-Union Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | false |
| `status` | `SOTA_CANDIDATE_UNION_NEEDS_REVIEW` |
| `scoped_claim_ready` | false |
| `strong_claim_ready` | false |
| `scenario_count` | 16 |
| `cache_source` | `default_cache+live_overlay:md/experiment_artifacts/service_cache_v2_live_merged_20260629.json` |
| `cache_record_count` | 290 |
| `union_makespan_beats_sota_envelope_all` | true |
| `union_mean_flow_beats_sota_envelope_all` | false |
| `guarded_union_not_pareto_dominated_all` | true |
| `single_guarded_union_beats_both_envelopes_all` | false |
| `adaptive_scalarized_union_not_pareto_dominated_all` | true |
| `adaptive_scalarized_union_beats_both_envelopes_all` | false |
| `pareto_slack_union_not_pareto_dominated_all` | true |
| `pareto_slack_union_beats_both_envelopes_all` | false |
| `fixed_online_policy_pareto_dominates_sota_style_all` | false |

## Scenario Envelope

| Taskset | Arrival | Best SOTA makespan | Pareto-slack makespan | Ratio | Best SOTA flow | Pareto-slack flow | Ratio | Pareto-slack dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | `static` | 8517.37 | 8517.37 | 1 | 4256.93 | 4256.93 | 1 | false |
| `q00_light_control` | `poisson` | 8533.21 | 8533.21 | 1 | 424.267 | 424.267 | 1 | false |
| `q01_gpu_bound_compute` | `static` | 1241.25 | 1241.25 | 1 | 646.783 | 646.783 | 1 | false |
| `q01_gpu_bound_compute` | `poisson` | 2501.87 | 2501.87 | 1 | 64.2812 | 64.2812 | 1 | false |
| `q01_gpu_bound_cnn_resnet50` | `static` | 47.6778 | 47.6778 | 1 | 27.0443 | 27.5362 | 0.982134 | false |
| `q01_gpu_bound_cnn_resnet50` | `poisson` | 3326.65 | 3326.65 | 1 | 2.50002 | 2.50002 | 1 | false |
| `q01_gpu_bound_llm_inference` | `static` | 3.60044 | 3.61573 | 0.995769 | 1.90269 | 1.9032 | 0.999733 | false |
| `q01_gpu_bound_llm_inference` | `poisson` | 2339.96 | 2340.06 | 0.999956 | 0.242715 | 0.344666 | 0.704203 | false |
| `q01_gpu_model_portfolio` | `static` | 829.79 | 829.79 | 1 | 86.5644 | 86.5114 | 1.00061 | false |
| `q01_gpu_model_portfolio` | `poisson` | 2681.92 | 2681.92 | 1 | 12.0576 | 12.1134 | 0.995393 | false |
| `q10_cpu_host_bound` | `static` | 4754.53 | 4754.53 | 1 | 2400.61 | 2400.61 | 1 | false |
| `q10_cpu_host_bound` | `poisson` | 4807.22 | 4807.22 | 1 | 569.804 | 569.804 | 1 | false |
| `q11_cpu_gpu_coupled` | `static` | 31381.6 | 31381.6 | 1 | 16059.2 | 16059.5 | 0.999983 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 31166.7 | 31166.7 | 1 | 14832.3 | 14832.3 | 1 | false |
| `hybrid_research_portfolio` | `static` | 23453.4 | 23453.4 | 1 | 4225.9 | 4226.68 | 0.999817 | false |
| `hybrid_research_portfolio` | `poisson` | 23471.4 | 23471.4 | 1 | 2957.02 | 2957.04 | 0.999996 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `legacy_fixed_caps` | `legacy` |  | 341169 | 16137.1 | 2.28882 | 5.15652 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149062 | 3129.32 | 1.00002 | 0.999954 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149212 | 3129.48 | 1.00103 | 1.00001 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149058 | 3129.47 | 0.999996 | 1 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149221 | 3129.56 | 1.00109 | 1.00003 |
| `scheduleurm_sota_union_online_pareto_slack` | `candidate` |  | 149059 | 3129.46 | 1 | 1 |
| `scheduleurm_sota_union_pareto_slack` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149058 | 3129.48 | 0.999997 | 1.00001 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis, Shockwave, Tiresias-style LAS | 149058 | 3129.49 | 0.999996 | 1.00001 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia, Optimus, AlloX | 149975 | 3130.36 | 1.00615 | 1.00029 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 149484 | 3129.51 | 1.00285 | 1.00002 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins, Tiresias-style LAS, AlloX/Optimus resource adaptation | 149975 | 3130.36 | 1.00615 | 1.00029 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva, AlloX | 149058 | 3129.49 | 0.999996 | 1.00001 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux, Optimus, AlloX | 149059 | 3129.48 | 1 | 1.00001 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 149970 | 3130.28 | 1.00612 | 1.00026 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family configuration-trajectory action-union dominance/envelope rows, not direct full-stack external binary superiority.
