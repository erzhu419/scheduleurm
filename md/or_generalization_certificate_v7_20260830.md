# OR Generalization Certificate v7

- Status: `OR_GENERALIZATION_V7_PASS_PROSPECTIVE_ACTION_UNIONS`
- Protocol validity, action-space closure, and universal superiority remain separate claims.

| Domain/evidence | Scope | Result | Retained boundary |
|---|---:|---|---|
| FJSP solver-action union | 20 prospective instances | exact union 20/20; same solver trajectory 20/20 | action-space closure, not CP-SAT superiority |
| MMRCPSP aggregate queue | 21 cells | baseline dominates 21/21 | negative result retained |
| Port statewise external holdout | 54 prospective instances | nondominated 27/54 | negative result retained |
| Port finite action union | 54 prospective instances | nondominated 54/54; strict-all 24/54 | registered family only |

## Claim boundary

- Supports: prospective 20-instance FJSP finite action-union evaluation.
- Supports: prospective 54-instance port statewise-policy counterexample.
- Supports: prospective 54-instance port finite action-union evaluation.
- Supports: prospective MMRCPSP structural and arrival-tape holdouts.
- Supports: finite-family selector portability across three scheduling domains.
- Does not support: global FJSP, MMRCPSP, or port optimality.
- Does not support: beating CP-SAT after its trajectory is admitted to the candidate family.
- Does not support: erasing the port statewise-policy or MMRCPSP aggregate-delay counterexamples.
- Does not support: universal delay, makespan, or source-objective superiority.
- Does not support: physical-port positive recurrence or industrial deployment.
