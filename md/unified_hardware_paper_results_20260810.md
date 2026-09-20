# Unified Hardware Paper Results

- Status: `PASS`
- Hardware scenarios: `16`
- Hardware traces: `525`
- Complete matrix runs: `16920`
- Registered online traces: `480`
- Online traces excluded from finite ratios because at least one policy did not complete: `3`
- Incomplete runs by policy: `{'ablation_delay_only': 3, 'ablation_high_profile_tiebreak': 3, 'ablation_no_bounded_penalty': 3, 'ablation_no_loaded_ledger': 3, 'ablation_no_migration': 3, 'ablation_support_only': 3, 'legacy_fixed_caps': 3, 'scheduleurm_unified_robust_maxweight': 3, 'sota_gavel_finish_time_fairness': 3, 'sota_gavel_pollux_sia_table_goodput': 3, 'sota_iadeep_salus_interference_guard': 3, 'sota_quadrant_composite': 3, 'sota_salus_iadeep_packing_guard': 3, 'sota_sia_pollux_resource_adaptive': 3, 'sota_srpt_gittins_mean_flow_oracle': 3}`
- Legacy/ours makespan geomean: `1.605662`
- Legacy/ours mean-flow geomean: `2.337904`
- SOTA non-dominance on every trace: `false`
- SOTA policy-trace comparisons: `3675`
- SOTA strict dominance of ours: `88`
- Ours strict dominance of SOTA: `2431`
- Migration comparisons: `36`
- Minimum hardware-local eta: `0.050000`

## SOTA-style policies

| Policy | Comparisons | Makespan baseline/ours | Mean flow baseline/ours | Baseline dominates count |
|---|---:|---:|---:|---:|
| `sota_gavel_finish_time_fairness` | 525 | 1.007814 | 1.028217 | 13 |
| `sota_gavel_pollux_sia_table_goodput` | 525 | 1.007814 | 1.028217 | 13 |
| `sota_iadeep_salus_interference_guard` | 525 | 1.010007 | 1.044527 | 6 |
| `sota_quadrant_composite` | 525 | 1.209911 | 1.239238 | 12 |
| `sota_salus_iadeep_packing_guard` | 525 | 1.226923 | 1.307439 | 15 |
| `sota_sia_pollux_resource_adaptive` | 525 | 1.014865 | 1.061179 | 14 |
| `sota_srpt_gittins_mean_flow_oracle` | 525 | 1.226784 | 1.292510 | 15 |

## SOTA-style trace-level dominance by quadrant

| Quadrant | Traces | Policy-trace comparisons | Baseline dominates | Ours dominates | Worst ours/baseline cost among baseline wins |
|---|---:|---:|---:|---:|---:|
| `q00` | 66 | 462 | 0 | 417 | 1.000000 |
| `q01` | 132 | 924 | 52 | 747 | 1.063766 |
| `q10` | 198 | 1386 | 4 | 934 | 1.077130 |
| `q11` | 129 | 903 | 32 | 333 | 1.012406 |

These are paper-facing summaries of the causal schema-v3 measured-cache policy-semantics population. Legacy is fixed-cap semantics on the same cache; external policies are not full-stack binary executions. Performance diagnostics are reported even when superiority is false and are never used as input-gate assumptions. Finite completion ratios use only traces on which all registered policies complete; incomplete tails are reported separately and these conditional ratios do not measure overall superiority.

## Static hardware-local legacy comparison

| Scenario | Traces | Makespan legacy/ours | Mean flow legacy/ours | P90 flow legacy/ours |
|---|---:|---:|---:|---:|
| `q00:node003_cpu_hpc_192c` | 3 | 4.083979 | 5.994441 | 5.808406 |
| `q00:node005_cpu_hpc_192c` | 3 | 4.087442 | 6.008666 | 5.792982 |
| `q01:gpu_2080_8gb_dual_jtl311linux` | 3 | 1.501980 | 2.364265 | 1.702092 |
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
