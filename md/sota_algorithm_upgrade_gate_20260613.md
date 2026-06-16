# SOTA Algorithm Upgrade Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `SOTA_ALGORITHM_UPGRADE_PASS` |
| `adaptive_scalarized_ready` | true |
| `pareto_slack_ready` | true |
| `fixed_online_policy_pareto_dominates_sota_style_all` | true |
| `state_dependent_marginal_cache_ready` | true |
| `lookahead_global_batch_ready` | true |
| `eta_reuse_ready` | true |
| `expanded_sota_families_ready` | true |

## Adaptive Scalarized Union

| Policy | Sum makespan | Weighted mean flow | vs best SOTA makespan envelope | vs best SOTA flow envelope |
|---|---:|---:|---:|---:|
| `scheduleurm_sota_union_adaptive_scalarized` | 152822 | 3171.39 | 1 | 0.999476 |

## Pareto-Slack Fixed Policy

| Policy | Sum makespan | Weighted mean flow | Worst vs best SOTA makespan | Worst vs best SOTA flow |
|---|---:|---:|---:|---:|
| `scheduleurm_sota_union_pareto_slack` | 152822 | 3171.39 | 1 | 0.999476 |

## Marginal Service States

| Workload | State | Certified | Relative to empty |
|---|---|---:|---:|
| `marginal_cuda` | `empty` | true | 1 |
| `marginal_cuda` | `high_vram_resident` | true | 1.09489 |
| `marginal_cpu` | `cpu_full_resident` | true | 1.00135 |
| `marginal_cpu` | `cpu_half_resident` | true | 1.09146 |
| `marginal_cpu` | `empty` | true | 1 |

## Scope

Replay/certificate evidence for an opt-in algorithm upgrade.  It admits SOTA-style action families into Scheduleurm's measured-cache candidate set and adds bounded lookahead/ETA reuse, but it is not a production launch trace and not direct full-stack SOTA superiority.
