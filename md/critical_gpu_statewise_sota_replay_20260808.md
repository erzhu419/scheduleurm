# Critical GPU Statewise SOTA Replay

- Status: `PASS`
- Performance validation: `True`

| Node | Scope | Arrival | Ours / best SOTA makespan | Ours / best SOTA flow | Strict all-SOTA | 0.5% dominance | 0.5% weak NI | Exact all-policy tie |
|---|---|---|---:|---:|:---:|:---:|:---:|:---:|
| `jtl110gpu` | `q01` | `static` | 1.001585 | 0.971876 | false | true | true | false |
| `jtl110gpu` | `q01` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |
| `jtl110gpu` | `q11` | `static` | 0.945959 | 0.934989 | true | true | true | false |
| `jtl110gpu` | `q11` | `poisson` | 0.972659 | 0.979162 | true | true | true | false |
| `jtl110gpu` | `portfolio` | `static` | 0.983840 | 0.960939 | true | true | true | false |
| `jtl110gpu` | `portfolio` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |
| `jtl110gpu2` | `q01` | `static` | 1.004584 | 0.972574 | false | true | true | false |
| `jtl110gpu2` | `q01` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |
| `jtl110gpu2` | `q11` | `static` | 0.798337 | 0.852540 | true | true | true | false |
| `jtl110gpu2` | `q11` | `poisson` | 0.947113 | 0.910882 | true | true | true | false |
| `jtl110gpu2` | `portfolio` | `static` | 0.789311 | 0.867985 | true | true | true | false |
| `jtl110gpu2` | `portfolio` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |
| `node007` | `q01` | `static` | 1.000000 | 0.992161 | true | true | true | false |
| `node007` | `q01` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |
| `node007` | `q11` | `static` | 0.962310 | 0.948276 | true | true | true | false |
| `node007` | `q11` | `poisson` | 1.000000 | 1.000000 | false | false | true | true |
| `node007` | `portfolio` | `static` | 0.966251 | 0.965173 | true | true | true | false |
| `node007` | `portfolio` | `poisson` | 1.000000 | 1.000000 | false | false | true | false |

## Performance requirements

- `candidate_not_pareto_dominated_in_any_scenario`: `true`
- `candidate_0p5pct_weakly_noninferior_in_every_scenario`: `true`
- `candidate_tolerance_dominates_all_sota_in_every_static_scenario`: `true`
- `strict_all_sota_improvement_observed_on_each_available_node`: `true`

## Diagnostics

- `candidate_not_pareto_dominated_in_any_scenario`: `true`
- `candidate_strictly_dominates_every_sota_policy_in_every_scenario`: `false`
- `candidate_0p5pct_tolerance_dominates_every_sota_policy_in_every_scenario`: `false`

Both Scheduleurm and every registered SOTA-style policy freeze their profile and event-level trajectory decisions on the same lower-service information set; the untouched natural-completion point view is used only for evaluation. PASS means no registered policy Pareto-dominates Scheduleurm in any of 18 scenarios, all comparisons are 0.5%-weakly-noninferior, every static scenario 0.5%-dominates every registered policy, and every measured node has at least one strict all-policy improvement. Sparse p1 ties are reported as ties, not strict wins. This is not a direct external binary comparison, does not pool the two RTX 3080 Ti hosts, and does not impute jtl311linux or unmeasured resource states.
