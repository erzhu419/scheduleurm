# BACASP-S factor-policy R85/R86 confirmation

- Status: `PORT_FACTOR_POLICY_HOLDOUT_PASS`
- Completed: `54/54`
- Pareto-nondominated: `16/54`
- Strict every-baseline dominance: `1/54`
- Distinct selected plans: `6`

| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |
|---|---|---:|---:|
| `fcfs_static` | `quay_makespan` | 0.627124 | [0.623287, 0.631366] |
| `fcfs_static` | `mean_quay_flow_time` | 0.227750 | [0.219928, 0.236104] |
| `fcfs_static` | `source_objective_cost` | 0.144206 | [0.135133, 0.154353] |
| `spt_static` | `quay_makespan` | 1.004678 | [1.002943, 1.006537] |
| `spt_static` | `mean_quay_flow_time` | 1.028667 | [1.018463, 1.039824] |
| `spt_static` | `source_objective_cost` | 1.047631 | [1.030603, 1.066211] |
| `edd_static` | `quay_makespan` | 0.627124 | [0.623366, 0.631274] |
| `edd_static` | `mean_quay_flow_time` | 0.234726 | [0.226607, 0.243445] |
| `edd_static` | `source_objective_cost` | 0.149396 | [0.140214, 0.159765] |
| `reconfiguration_greedy` | `quay_makespan` | 1.004678 | [1.002978, 1.006544] |
| `reconfiguration_greedy` | `mean_quay_flow_time` | 1.028667 | [1.018789, 1.039718] |
| `reconfiguration_greedy` | `source_objective_cost` | 1.047631 | [1.031090, 1.066226] |

The confirmation is limited to the registered BACASP-S fixed-slot source
core. It is not an unrestricted berth-allocation optimum, physical-terminal
deployment, migration experiment, or stochastic-stability certificate.
