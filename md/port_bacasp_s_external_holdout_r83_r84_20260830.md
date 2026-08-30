# BACASP-S R83/R84 external holdout

- Status: `PORT_PUBLIC_EXTERNAL_HOLDOUT_PASS`
- Instances: `54/54`.
- Pareto-nondominated: `27/54`.
- Strictly dominates every registered baseline: `12/54`.
- Frozen plan: `plan|spt_static+robust_maxweight+robust_maxweight`.
- Full gzip SHA256: `0d1d3a753c78a50684f6fb2336ad60358baec60a864f3e9e02a0413244254add`.

| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |
|---|---|---:|---:|
| `fcfs_static` | `quay_makespan` | 0.592819 | [0.583147, 0.603302] |
| `fcfs_static` | `mean_quay_flow_time` | 0.292148 | [0.280559, 0.304572] |
| `fcfs_static` | `source_objective_cost` | 0.220045 | [0.207684, 0.234590] |
| `spt_static` | `quay_makespan` | 0.995375 | [0.989489, 1.001544] |
| `spt_static` | `mean_quay_flow_time` | 1.040881 | [1.031057, 1.050472] |
| `spt_static` | `source_objective_cost` | 1.053769 | [1.038296, 1.068792] |
| `edd_static` | `quay_makespan` | 0.592452 | [0.583004, 0.603202] |
| `edd_static` | `mean_quay_flow_time` | 0.299922 | [0.287792, 0.312790] |
| `edd_static` | `source_objective_cost` | 0.226892 | [0.214024, 0.241082] |
| `reconfiguration_greedy` | `quay_makespan` | 0.995363 | [0.989434, 1.001280] |
| `reconfiguration_greedy` | `mean_quay_flow_time` | 1.040861 | [1.031201, 1.050653] |
| `reconfiguration_greedy` | `source_objective_cost` | 1.053662 | [1.039015, 1.068844] |

This is a fixed-slot, source-core, post-freeze public holdout. It is not
a continuous-quay optimum, author-algorithm comparison, physical-port
deployment, or port stochastic-stability certificate.
