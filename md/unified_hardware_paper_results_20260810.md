# Unified Hardware Paper Results

- Status: `PASS`
- Hardware scenarios: `16`
- Hardware traces: `528`
- Complete matrix runs: `16920`
- Legacy/ours makespan geomean: `1.638137`
- Legacy/ours mean-flow geomean: `2.384272`
- SOTA non-dominance on every trace: `false`
- SOTA policy-trace comparisons: `3696`
- SOTA strict dominance of ours: `71`
- Ours strict dominance of SOTA: `2446`
- Migration comparisons: `36`
- Minimum hardware-local eta: `0.050000`

## SOTA-style policies

| Policy | Comparisons | Makespan baseline/ours | Mean flow baseline/ours | Baseline dominates count |
|---|---:|---:|---:|---:|
| `sota_gavel_finish_time_fairness` | 528 | 1.013288 | 1.030963 | 12 |
| `sota_gavel_pollux_sia_table_goodput` | 528 | 1.013288 | 1.030963 | 12 |
| `sota_iadeep_salus_interference_guard` | 528 | 1.019173 | 1.051119 | 6 |
| `sota_quadrant_composite` | 528 | 1.215221 | 1.241231 | 11 |
| `sota_salus_iadeep_packing_guard` | 528 | 1.281170 | 1.326607 | 12 |
| `sota_sia_pollux_resource_adaptive` | 528 | 1.036710 | 1.053084 | 9 |
| `sota_srpt_gittins_mean_flow_oracle` | 528 | 1.281402 | 1.311431 | 9 |

## SOTA-style trace-level dominance by quadrant

| Quadrant | Traces | Policy-trace comparisons | Baseline dominates | Ours dominates | Worst ours/baseline cost among baseline wins |
|---|---:|---:|---:|---:|---:|
| `q00` | 66 | 462 | 0 | 417 | 1.000000 |
| `q01` | 132 | 924 | 35 | 755 | 1.059856 |
| `q10` | 198 | 1386 | 4 | 934 | 1.077130 |
| `q11` | 132 | 924 | 32 | 340 | 1.012406 |

These are paper-facing summaries of the exact schema-v2 measured-cache policy-semantics population. Legacy is fixed-cap semantics on the same cache; external policies are not full-stack binary executions. Performance diagnostics are reported even when superiority is false and are never used as input-gate assumptions.

## Static hardware-local legacy comparison

| Scenario | Traces | Makespan legacy/ours | Mean flow legacy/ours | P90 flow legacy/ours |
|---|---:|---:|---:|---:|
| `q00:node003_cpu_hpc_192c` | 3 | 4.083979 | 5.994441 | 5.808406 |
| `q00:node005_cpu_hpc_192c` | 3 | 4.087442 | 6.008666 | 5.792982 |
| `q01:gpu_2080_8gb_dual_jtl311linux` | 3 | 1.527841 | 2.367677 | 1.702092 |
| `q01:gpu_2080ti_11gb_quad_node007` | 3 | 1.153703 | 1.532013 | 1.273425 |
| `q01:gpu_3080ti_12gb_dual_jtl110gpu` | 3 | 1.207497 | 1.564309 | 1.312066 |
| `q01:gpu_3080ti_12gb_dual_jtl110gpu2` | 3 | 1.132329 | 1.499733 | 1.192372 |
| `q10:node001_cpu_hpc_192c` | 3 | 2.422221 | 1.705248 | 2.301150 |
| `q10:node002_cpu_hpc_192c` | 3 | 2.430438 | 1.698841 | 2.301980 |
| `q10:node003_cpu_hpc_192c` | 3 | 3.398961 | 3.486260 | 3.421811 |
| `q10:node004_cpu_hpc_192c` | 3 | 3.607959 | 3.713465 | 3.644578 |
| `q10:node005_cpu_hpc_192c` | 3 | 3.398778 | 3.476316 | 3.420893 |
| `q10:node006_cpu_hpc_192c` | 3 | 3.591276 | 3.692842 | 3.628476 |
| `q11:gpu_2080_8gb_dual_jtl311linux` | 3 | 1.027136 | 0.994510 | 0.900864 |
| `q11:gpu_2080ti_11gb_quad_node007` | 3 | 1.123543 | 1.217034 | 1.093424 |
| `q11:gpu_3080ti_12gb_dual_jtl110gpu` | 3 | 1.030952 | 0.995402 | 1.026782 |
| `q11:gpu_3080ti_12gb_dual_jtl110gpu2` | 3 | 1.035129 | 0.998460 | 1.024511 |
