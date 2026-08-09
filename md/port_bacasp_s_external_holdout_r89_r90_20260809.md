# BACASP-S R89/R90 external holdout

- Status: `PORT_PUBLIC_EXTERNAL_HOLDOUT_FAIL`
- Instances: `36/54`.
- Pareto-nondominated: `15/54`.
- Strictly dominates every registered baseline: `12/54`.
- Frozen plan: `plan|spt_static+robust_maxweight+robust_maxweight`.
- Full gzip SHA256: `11b77d9e8220474030167f1355e1dc156fdcf33a26719c4df6e99d1408ce77fd`.

| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |
|---|---|---:|---:|
| `fcfs_static` | `quay_makespan` | 0.611718 | [0.609097, 0.614497] |
| `fcfs_static` | `mean_quay_flow_time` | 0.266074 | [0.263684, 0.268540] |
| `fcfs_static` | `source_objective_cost` | 0.168306 | [0.160080, 0.176669] |
| `spt_static` | `quay_makespan` | 1.004752 | [1.000902, 1.009020] |
| `spt_static` | `mean_quay_flow_time` | 1.002282 | [0.995382, 1.009858] |
| `spt_static` | `source_objective_cost` | 0.985902 | [0.966814, 1.006126] |
| `edd_static` | `quay_makespan` | 0.611730 | [0.609115, 0.614531] |
| `edd_static` | `mean_quay_flow_time` | 0.269207 | [0.266620, 0.271843] |
| `edd_static` | `source_objective_cost` | 0.170580 | [0.162604, 0.178468] |
| `reconfiguration_greedy` | `quay_makespan` | 1.004752 | [1.000859, 1.009054] |
| `reconfiguration_greedy` | `mean_quay_flow_time` | 1.002282 | [0.995131, 1.009521] |
| `reconfiguration_greedy` | `source_objective_cost` | 0.985902 | [0.966504, 1.005961] |

This is a fixed-slot, source-core, post-freeze public holdout. It is not
a continuous-quay optimum, author-algorithm comparison, physical-port
deployment, or port stochastic-stability certificate.
