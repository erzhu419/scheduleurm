# BACASP-S R87/R88 external holdout

- Status: `PORT_PUBLIC_EXTERNAL_HOLDOUT_PASS`
- Instances: `54/54`.
- Pareto-nondominated: `12/54`.
- Strictly dominates every registered baseline: `3/54`.
- Frozen plan: `plan|spt_static+robust_maxweight+robust_maxweight`.
- Full gzip SHA256: `3bcd44a0ada09de8e33adce7aed5aecf25950805a48bf93fe24140b8d34fe72a`.

| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |
|---|---|---:|---:|
| `edd_static` | `mean_quay_flow_time` | 0.279001 | [0.266679, 0.291937] |
| `edd_static` | `quay_makespan` | 0.609741 | [0.600875, 0.619411] |
| `edd_static` | `source_objective_cost` | 0.192736 | [0.179364, 0.207384] |
| `fcfs_static` | `mean_quay_flow_time` | 0.275264 | [0.263262, 0.288135] |
| `fcfs_static` | `quay_makespan` | 0.609710 | [0.600637, 0.619683] |
| `fcfs_static` | `source_objective_cost` | 0.189606 | [0.176822, 0.204370] |
| `reconfiguration_greedy` | `mean_quay_flow_time` | 1.028179 | [1.020212, 1.035918] |
| `reconfiguration_greedy` | `quay_makespan` | 1.007921 | [1.004668, 1.011138] |
| `reconfiguration_greedy` | `source_objective_cost` | 1.044452 | [1.028220, 1.059544] |
| `spt_static` | `mean_quay_flow_time` | 1.028179 | [1.019961, 1.035982] |
| `spt_static` | `quay_makespan` | 1.007921 | [1.004783, 1.011223] |
| `spt_static` | `source_objective_cost` | 1.044452 | [1.028540, 1.059645] |

This is a fixed-slot, source-core, post-freeze public holdout. It is not
a continuous-quay optimum, author-algorithm comparison, physical-port
deployment, or port stochastic-stability certificate.
