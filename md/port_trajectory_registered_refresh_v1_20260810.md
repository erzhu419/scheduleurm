# Port trajectory registered refresh v1

- Status: `PORT_TRAJECTORY_REGISTERED_REFRESH_PASS`
- Chronology: registered retrospective replay; this is not a prospective holdout.
- Full semantic ledger hash: `77a8095148e7f087347888be2d81279d62f05cb4ff63bd88aab70063787efc15`
- Candidate/context evaluations: `124`.
- Generated-family oracle gap: `0.0`.

| Scope | Registered evidence | Result |
|---|---:|---|
| Public BACASP-S source core | 1 context | ours Pareto nondominated: `true` |
| Synthetic berth/quay/yard/gate | 3 contexts | ours Pareto nondominated 3/3 |
| Reconfiguration family | 3 analogues | `crane_reassignment`, `reberth`, `yard_rehandle` |

## Claim boundary

- Supports: current-code deterministic reproduction of the registered 31-candidate trajectory family.
- Supports: one locked global plan over one public and three synthetic port contexts.
- Supports: exact generated-family selection and feasibility for all 124 trajectory-context pairs.
- Supports: registered four-resource re-berthing, crane-reassignment, and yard-rehandle semantics.
- Does not support: a prospective port holdout.
- Does not support: measured operation or safety at a physical terminal.
- Does not support: unrestricted port optimality or universal policy superiority.
- Does not support: stochastic port stability from deterministic replay alone.
