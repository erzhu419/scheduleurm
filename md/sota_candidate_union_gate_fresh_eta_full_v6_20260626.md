# SOTA Candidate-Union Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `SOTA_CANDIDATE_UNION_PASS` |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `scenario_count` | 16 |
| `union_makespan_beats_sota_envelope_all` | false |
| `union_mean_flow_beats_sota_envelope_all` | true |
| `guarded_union_not_pareto_dominated_all` | true |
| `single_guarded_union_beats_both_envelopes_all` | false |
| `adaptive_scalarized_union_not_pareto_dominated_all` | true |
| `adaptive_scalarized_union_beats_both_envelopes_all` | false |
| `pareto_slack_union_not_pareto_dominated_all` | true |
| `pareto_slack_union_beats_both_envelopes_all` | false |
| `fixed_online_policy_pareto_dominates_sota_style_all` | true |

## Scenario Envelope

| Taskset | Arrival | Best SOTA makespan | Pareto-slack makespan | Ratio | Best SOTA flow | Pareto-slack flow | Ratio | Pareto-slack dominated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | `static` | 8517.37 | 8517.37 | 1 | 4256.93 | 4256.93 | 1 | false |
| `q00_light_control` | `poisson` | 8533.21 | 8533.21 | 1 | 424.267 | 424.267 | 1 | false |
| `q01_gpu_bound_compute` | `static` | 1241.25 | 1241.25 | 1 | 646.783 | 646.783 | 1 | false |
| `q01_gpu_bound_compute` | `poisson` | 2501.87 | 2501.87 | 1 | 64.2812 | 64.2812 | 1 | false |
| `q01_gpu_bound_cnn_resnet50` | `static` | 47.5468 | 47.8699 | 0.993251 | 26.9713 | 26.9713 | 1 | false |
| `q01_gpu_bound_cnn_resnet50` | `poisson` | 3326.65 | 3326.65 | 1 | 2.50118 | 2.50118 | 1 | false |
| `q01_gpu_bound_llm_inference` | `static` | 3.60623 | 3.61573 | 0.99737 | 1.90309 | 1.9032 | 0.999943 | false |
| `q01_gpu_bound_llm_inference` | `poisson` | 2339.94 | 2340.06 | 0.999949 | 0.224229 | 0.344666 | 0.65057 | false |
| `q01_gpu_model_portfolio` | `static` | 829.79 | 829.79 | 1 | 86.5613 | 86.5616 | 0.999997 | false |
| `q01_gpu_model_portfolio` | `poisson` | 2681.92 | 2681.92 | 1 | 12.0478 | 12.1137 | 0.994558 | false |
| `q10_cpu_host_bound` | `static` | 4754.53 | 4754.53 | 1 | 2400.61 | 2400.61 | 1 | false |
| `q10_cpu_host_bound` | `poisson` | 4807.22 | 4807.22 | 1 | 569.804 | 569.804 | 1 | false |
| `q11_cpu_gpu_coupled` | `static` | 31368.8 | 31368.8 | 1 | 16059.4 | 16059.4 | 1 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 31141.7 | 31141.7 | 1 | 14832.5 | 14832.5 | 1 | false |
| `hybrid_research_portfolio` | `static` | 23358.9 | 23358.9 | 1 | 4227.43 | 4227.43 | 1 | false |
| `hybrid_research_portfolio` | `poisson` | 23472.2 | 23472.2 | 1 | 2957.02 | 2957.03 | 0.999995 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `legacy_fixed_caps` | `legacy` |  | 341169 | 16137.1 | 2.29086 | 5.15635 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148962 | 3129.38 | 1.00024 | 0.99994 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149584 | 3129.7 | 1.00442 | 1.00004 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148962 | 3129.36 | 1.00024 | 0.999937 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149555 | 3129.95 | 1.00422 | 1.00013 |
| `scheduleurm_sota_union_online_pareto_slack` | `candidate` |  | 148926 | 3129.56 | 1 | 1 |
| `scheduleurm_sota_union_pareto_slack` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148927 | 3129.58 | 1 | 1 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis, Shockwave, Tiresias-style LAS | 148930 | 3129.58 | 1.00002 | 1 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia, Optimus, AlloX | 149940 | 3130.51 | 1.00681 | 1.0003 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 150665 | 3132.94 | 1.01167 | 1.00108 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins, Tiresias-style LAS, AlloX/Optimus resource adaptation | 149940 | 3130.51 | 1.00681 | 1.0003 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva, AlloX | 148930 | 3129.58 | 1.00002 | 1 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux, Optimus, AlloX | 148927 | 3129.57 | 1 | 1 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 149940 | 3130.51 | 1.00681 | 1.0003 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family configuration-trajectory action-union dominance/envelope rows, not direct full-stack external binary superiority.
