# SOTA Candidate-Union Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `SOTA_CANDIDATE_UNION_PASS` |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `scenario_count` | 16 |
| `cache_source` | `default_cache+live_overlay:md/experiment_artifacts/service_cache_v2_live_merged_20260701.json` |
| `cache_record_count` | 689 |
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
| `q11_cpu_gpu_coupled` | `static` | 31372.5 | 31372.5 | 1 | 16059.4 | 16059.4 | 1 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 31171.6 | 31171.6 | 1 | 14831.8 | 14831.8 | 1 | false |
| `hybrid_research_portfolio` | `static` | 23365.3 | 23365.3 | 1 | 4227.48 | 4227.48 | 1 | false |
| `hybrid_research_portfolio` | `poisson` | 23461.9 | 23461.9 | 1 | 2957.77 | 2957.78 | 0.999995 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `legacy_fixed_caps` | `legacy` |  | 341169 | 16137.1 | 2.2904 | 5.15622 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149047 | 3129.48 | 1.00061 | 0.999947 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149004 | 3129.39 | 1.00032 | 0.999921 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 149047 | 3129.46 | 1.00061 | 0.999943 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148956 | 3129.65 | 1 | 1 |
| `scheduleurm_sota_union_online_pareto_slack` | `candidate` |  | 148956 | 3129.64 | 1 | 1 |
| `scheduleurm_sota_union_pareto_slack` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 148957 | 3129.66 | 1 | 1 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis, Shockwave, Tiresias-style LAS | 148959 | 3129.66 | 1.00002 | 1 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia, Optimus, AlloX | 149940 | 3130.51 | 1.00661 | 1.00028 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 149519 | 3130.38 | 1.00378 | 1.00024 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins, Tiresias-style LAS, AlloX/Optimus resource adaptation | 149940 | 3130.51 | 1.00661 | 1.00028 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva, AlloX | 148959 | 3129.66 | 1.00002 | 1 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux, Optimus, AlloX | 148956 | 3129.65 | 1 | 1 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 149940 | 3130.51 | 1.00661 | 1.00028 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family configuration-trajectory action-union dominance/envelope rows, not direct full-stack external binary superiority.
