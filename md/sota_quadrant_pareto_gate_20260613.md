# SOTA Four-Quadrant Pareto Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `SOTA_QUADRANT_STRICT_NONINFERIOR_PASS` |
| `tolerance_pareto_ready` | true |
| `strict_noninferiority_ready` | true |
| `strict_pareto_dominance_ready` | false |
| `pareto_tolerance` | 0.005 |
| `strict_frontier_count` | 0 |

## Quadrants

| Quadrant | Scenarios | Worst makespan ratio | Worst flow ratio | Strict noninferior | Strict improvement | Strict dominate |
|---|---:|---:|---:|---:|---:|---:|
| `q00_low_cpu_low_gpu` | 2 | 1 | 1 | true | false | false |
| `q01_low_cpu_high_gpu` | 8 | 1 | 1 | true | true | true |
| `q10_high_cpu_low_gpu` | 2 | 1 | 1 | true | false | false |
| `q11_high_cpu_high_gpu` | 2 | 1 | 1 | true | false | false |
| `hybrid_portfolio` | 2 | 1 | 1 | true | true | true |

## Scope

Four-quadrant measured-cache policy-semantics gate.  Strict dominance is reported separately from the 0.5% replay-tolerance Pareto claim.
