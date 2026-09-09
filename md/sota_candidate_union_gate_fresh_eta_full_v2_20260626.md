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
| `single_guarded_union_beats_both_envelopes_all` | true |
| `adaptive_scalarized_union_not_pareto_dominated_all` | true |
| `adaptive_scalarized_union_beats_both_envelopes_all` | true |
| `pareto_slack_union_not_pareto_dominated_all` | true |
| `pareto_slack_union_beats_both_envelopes_all` | true |
| `fixed_online_policy_pareto_dominates_sota_style_all` | true |

## Scenario Envelope

| Taskset | Arrival | Best SOTA makespan | Pareto-slack makespan | Ratio | Best SOTA flow | Pareto-slack flow | Ratio | Pareto-slack dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | `static` | 8517.37 | 8517.37 | 1 | 4256.93 | 4256.93 | 1 | false |
| `q00_light_control` | `poisson` | 8533.21 | 8533.21 | 1 | 424.267 | 424.267 | 1 | false |
| `q01_gpu_bound_compute` | `static` | 1241.25 | 1241.25 | 1 | 646.783 | 646.783 | 1 | false |
| `q01_gpu_bound_compute` | `poisson` | 2501.87 | 2501.87 | 1 | 64.2812 | 64.2812 | 1 | false |
| `q01_gpu_bound_cnn_resnet50` | `static` | 50.7545 | 50.7545 | 1 | 27.44 | 27.4544 | 0.999476 | false |
| `q01_gpu_bound_cnn_resnet50` | `poisson` | 3326.88 | 3326.88 | 1 | 2.72852 | 2.72852 | 1 | false |
| `q01_gpu_bound_llm_inference` | `static` | 2.06026 | 2.06026 | 1 | 1.10738 | 1.10738 | 1 | false |
| `q01_gpu_bound_llm_inference` | `poisson` | 2339.93 | 2339.93 | 1 | 0.210106 | 0.210106 | 1 | false |
| `q01_gpu_model_portfolio` | `static` | 829.79 | 829.79 | 1 | 86.317 | 86.3054 | 1.00013 | false |
| `q01_gpu_model_portfolio` | `poisson` | 2682.16 | 2682.16 | 1 | 12.102 | 12.102 | 1 | false |
| `q10_cpu_host_bound` | `static` | 4754.53 | 4754.53 | 1 | 2400.61 | 2400.61 | 1 | false |
| `q10_cpu_host_bound` | `poisson` | 4807.22 | 4807.22 | 1 | 569.804 | 569.804 | 1 | false |
| `q11_cpu_gpu_coupled` | `static` | 31368.8 | 31368.8 | 1 | 16059.4 | 16059.4 | 1 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 31141.7 | 31141.7 | 1 | 14832.5 | 14832.5 | 1 | false |
| `hybrid_research_portfolio` | `static` | 23441.3 | 23441.3 | 1 | 4226.76 | 4226.76 | 1 | false |
| `hybrid_research_portfolio` | `poisson` | 23447.3 | 23447.3 | 1 | 2957.32 | 2957.32 | 1 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `calibrated_backlog_aware_guarded` | `candidate` |  | 150199 | 3130.23 | 1 | 1 |
| `legacy_fixed_caps` | `legacy` |  | 341279 | 16137.2 | 2.27219 | 5.15527 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148986 | 3129.48 | 0.991927 | 0.999761 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148986 | 3129.48 | 0.991927 | 0.999761 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148986 | 3129.48 | 0.991927 | 0.999761 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 150353 | 3132.21 | 1.00103 | 1.00063 |
| `scheduleurm_sota_union_pareto_slack` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148986 | 3129.49 | 0.991928 | 0.999762 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis, Shockwave, Tiresias-style LAS | 148986 | 3129.49 | 0.991928 | 0.999762 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia, Optimus, AlloX | 150279 | 3130.5 | 1.00054 | 1.00009 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 150354 | 3132.21 | 1.00104 | 1.00063 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins, Tiresias-style LAS, AlloX/Optimus resource adaptation | 150279 | 3130.5 | 1.00054 | 1.00009 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva, AlloX | 148986 | 3129.49 | 0.991928 | 0.999762 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux, Optimus, AlloX | 148986 | 3129.49 | 0.991928 | 0.999762 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 150279 | 3130.5 | 1.00054 | 1.00009 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family configuration-trajectory action-union dominance/envelope rows, not direct full-stack external binary superiority.
