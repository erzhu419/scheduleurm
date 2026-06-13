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
| `q01_gpu_model_portfolio` | `static` | 829.79 | 829.79 | 1 | 86.317 | 86.3183 | 0.999985 | false |
| `q01_gpu_model_portfolio` | `poisson` | 2682.16 | 2682.16 | 1 | 12.102 | 12.102 | 1 | false |
| `q10_cpu_host_bound` | `static` | 4754.53 | 4754.53 | 1 | 2400.61 | 2400.61 | 1 | false |
| `q10_cpu_host_bound` | `poisson` | 4807.22 | 4807.22 | 1 | 569.804 | 569.804 | 1 | false |
| `q11_cpu_gpu_coupled` | `static` | 38336.4 | 38382.4 | 0.998801 | 19394.4 | 19394.4 | 1 | false |
| `q11_cpu_gpu_coupled` | `poisson` | 38135.9 | 38184.5 | 0.998727 | 18145.7 | 18145.7 | 1 | false |
| `hybrid_research_portfolio` | `static` | 28508.8 | 28508.8 | 1 | 4819.38 | 4834.47 | 0.996877 | false |
| `hybrid_research_portfolio` | `poisson` | 28552.9 | 28552.9 | 1 | 3541.54 | 3556.53 | 0.995784 | false |

## Aggregate

| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---|---:|---:|---:|---:|
| `calibrated_backlog_aware_guarded` | `candidate` |  | 173859 | 3569.97 | 1 | 1 |
| `legacy_fixed_caps` | `legacy` |  | 388938 | 17038.5 | 2.23709 | 4.77274 |
| `scheduleurm_sota_union_adaptive_scalarized` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173487 | 3569.5 | 0.997858 | 0.999869 |
| `scheduleurm_sota_union_guarded_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173487 | 3569.5 | 0.997858 | 0.999869 |
| `scheduleurm_sota_union_makespan` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173216 | 3573.49 | 0.996298 | 1.00099 |
| `scheduleurm_sota_union_mean_flow` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173487 | 3569.5 | 0.997858 | 0.999869 |
| `scheduleurm_sota_union_pareto_slack` | `candidate_union` | Gavel, Pollux, Sia, IADeep, Salus, SRPT, Gittins | 173216 | 3573.49 | 0.996298 | 1.00099 |
| `sota_gavel_finish_time_fairness` | `sota_style` | Gavel, Themis | 173487 | 3569.5 | 0.997858 | 0.999869 |
| `sota_gavel_pollux_sia_table_goodput` | `sota_style` | Gavel, Pollux, Sia | 173592 | 3574.06 | 0.998462 | 1.00114 |
| `sota_iadeep_salus_interference_guard` | `sota_style` | IADeep, Salus | 173336 | 3569.82 | 0.996987 | 0.999957 |
| `sota_quadrant_composite` | `sota_style` | Gavel/Pollux/Sia, IADeep/Salus, SRPT/Gittins | 173859 | 3569.97 | 1 | 1 |
| `sota_salus_iadeep_packing_guard` | `sota_style` | Salus, IADeep, Gandiva | 173216 | 3573.49 | 0.996298 | 1.00099 |
| `sota_sia_pollux_resource_adaptive` | `sota_style` | Sia, Pollux | 173487 | 3569.5 | 0.997858 | 0.999869 |
| `sota_srpt_gittins_mean_flow_oracle` | `sota_style` | SRPT, Gittins, SERPT | 173859 | 3569.97 | 1 | 1 |

## Scope

Policy-semantics gate on identical measured service curves.  It certifies SOTA-policy-family action-union dominance/envelope rows, not direct full-stack external binary superiority.
