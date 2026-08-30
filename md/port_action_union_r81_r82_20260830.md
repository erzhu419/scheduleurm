# BACASP-S R81/R82 finite action-union holdout

- Status: `PORT_ACTION_UNION_HOLDOUT_PASS`
- Instances: `54/54`.
- Pareto-nondominated: `54/54`.
- Strictly dominates every baseline: `24/54`.
- Candidate plans: `11`.

| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |
|---|---|---:|---:|
| `fcfs_static` | `quay_makespan` | 0.574370 | [0.570100, 0.578949] |
| `fcfs_static` | `mean_quay_flow_time` | 0.266716 | [0.261557, 0.272083] |
| `fcfs_static` | `source_objective_cost` | 0.198934 | [0.191729, 0.206597] |
| `spt_static` | `quay_makespan` | 0.988592 | [0.986239, 0.991011] |
| `spt_static` | `mean_quay_flow_time` | 0.965929 | [0.958475, 0.973289] |
| `spt_static` | `source_objective_cost` | 0.945591 | [0.933598, 0.957517] |
| `edd_static` | `quay_makespan` | 0.574360 | [0.570188, 0.578830] |
| `edd_static` | `mean_quay_flow_time` | 0.272577 | [0.267132, 0.278203] |
| `edd_static` | `source_objective_cost` | 0.203905 | [0.196870, 0.211455] |
| `reconfiguration_greedy` | `quay_makespan` | 0.988592 | [0.986300, 0.990980] |
| `reconfiguration_greedy` | `mean_quay_flow_time` | 0.965929 | [0.958444, 0.973320] |
| `reconfiguration_greedy` | `source_objective_cost` | 0.945591 | [0.933935, 0.957675] |

The action union is exact only over the registered deterministic candidate
family. It is not an unrestricted BACASP-S optimum, author-algorithm
comparison, physical-port deployment, or stochastic-stability certificate.
