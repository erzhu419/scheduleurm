# FJSP Numeric Disposition Gate

- Status: `FJSP_NUMERIC_DISPOSITION_PASS`
- Frozen source artifacts rewritten: `false`
- Comparison coordinates: `(makespan_ticks, sum_completion_ticks)`

| Holdout | Old nondominated | Exact-ledger nondominated | False dominance |
|---|---:|---:|---:|
| `first_family_holdout` | 19/20 | 20/20 | 1 |
| `hurink_confirmation` | 19/20 | 20/20 | 1 |

## Dispositions

- `first_family_holdout:ChambersBarnes1996/setb4xx.txt`: old dominator(s) `ours_robust_maxweight` have the same exact schedule fingerprint and are reclassified as `same_action`.
- `hurink_confirmation:vdata/la10.txt`: old dominator(s) `most_work_remaining` have the same exact schedule fingerprint and are reclassified as `same_action`.

This disposition corrects only two baseline-union Pareto relations whose stored schedules are identical but whose mean-flow values were rounded at different precisions. It does not alter either prospective source artifact and does not remove the separate CP-SAT candidate-family gap.
