# Statewise Phase-Aware CPU SOTA Replay

- Status: `PASS`
- Node: `node001`
- Profiles: `[1, 2, 4]`

| Check | Pass |
|---|---:|
| `exact_projection_row_count` | true |
| `all_exact_rows_natural_completion_ready` | true |
| `no_history_fallback` | true |
| `separate_lower_and_completion_views` | true |
| `all_scenarios_complete` | true |
| `candidate_selection_bound_to_lower_service` | true |

| Arrival | Ours / best SOTA makespan | Ours / best SOTA mean flow | Pareto dominated |
|---|---:|---:|---:|
| `static` | 1.000000 | 1.000000 | false |
| `poisson` | 1.000000 | 1.000000 | false |

PASS certifies a fail-closed, phase-aware SOTA-policy replay for the exact native FreqDuet/SUMO rows on the declared CPU node and profiles. Scheduleurm actions are selected on lower service and evaluated on natural-completion point service. It does not cover GPU profiles, other load states, unregistered workloads, or direct external scheduler binaries. Performance ratios are reported, not used as infrastructure-gate pass conditions.
