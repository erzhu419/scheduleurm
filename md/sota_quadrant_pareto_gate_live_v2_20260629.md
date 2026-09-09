# SOTA Four-Quadrant Pareto Gate

| Quantity | Value |
|---|---:|
| `pass` | false |
| `status` | `SOTA_QUADRANT_TOLERANCE_OPEN` |
| `tolerance_pareto_ready` | false |
| `strict_noninferiority_ready` | false |
| `strict_pareto_dominance_ready` | false |
| `pareto_tolerance` | 0.005 |
| `strict_frontier_count` | 3 |

## Quadrants

| Quadrant | Scenarios | Worst makespan ratio | Worst flow ratio | Strict noninferior | Strict improvement | Strict dominate |
|---|---:|---:|---:|---:|---:|---:|
| `q00_low_cpu_low_gpu` | 2 | 1 | 1 | true | false | false |
| `q01_low_cpu_high_gpu` | 8 | 0.988708875 | 1 | false | true | false |
| `q10_high_cpu_low_gpu` | 2 | 1 | 1 | true | false | false |
| `q11_high_cpu_high_gpu` | 2 | 1 | 0.999983343 | false | false | false |
| `hybrid_portfolio` | 2 | 1 | 0.999817413 | false | false | false |

## Scope

Four-quadrant measured-cache policy-semantics gate.  Strict dominance is reported separately from the 0.5% replay-tolerance Pareto claim.
