# FJSP exact-ledger replication audit

- Status: `FJSP_REPLICATION_AUDIT_PASS_WITH_CP_SAT_VARIATION`
- Candidate ledger identity: `20/20`
- CP-SAT qualitative disposition identity: `20/20`
- CP-SAT numeric gap identity: `17/20`
- Strict full-artifact reproducibility: `false`

The frozen Scheduleurm candidate ledgers reproduce exactly. The 5-second single-worker CP-SAT comparator preserves feasibility, optimal-status counts, and qualitative Pareto relations, but its incumbent gaps are not bitwise stable.

| Instance | Primary gap | Replication gap |
|---|---|---|
| edata/la26.txt | `{'makespan_ticks': 90, 'sum_completion_ticks': 1722}` | `{'makespan_ticks': 86, 'sum_completion_ticks': 1377}` |
| rdata/abz8.txt | `{'makespan_ticks': 29, 'sum_completion_ticks': 622}` | `{'makespan_ticks': 32, 'sum_completion_ticks': 582}` |
| sdata/la26.txt | `{'makespan_ticks': 201, 'sum_completion_ticks': 1722}` | `{'makespan_ticks': 206, 'sum_completion_ticks': 2323}` |

## Claim Boundary

{'supports': 'repeatability of the frozen candidate action ledgers and the qualitative fixed-budget comparator dispositions over these two runs', 'does_not_support': 'bitwise reproducibility of a wall-clock-limited CP-SAT incumbent or global FJSP optimality', 'interpretation': 'The 5-second single-worker CP-SAT reference preserves status and Pareto relation here, but three incumbent objective gaps vary.'}
