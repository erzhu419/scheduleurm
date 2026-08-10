# Port statewise trajectory v3

- Status: `PORT_STATEWISE_TRAJECTORY_V3_PASS`
- Artifact SHA-256: `bc896d43e89fea6dd8a24017f98d33adacd1baaebc7c7661aa9ff58586a82537`
- BACASP source-core ready: `true`
- Synthetic statewise migration ready: `true`
- Exact duration-normalized oracle gap: `0.0`
- Finite-family penalty bound P0: `2.54`
- Selected action: `robust_delayed_reberth`
- Terminal-cost Pareto status is diagnostic and never overrides the robust oracle.
- BACASP source-core evidence and synthetic migration evidence are non-substitutable.

## Claim Boundary

Supported:
- a finite statewise candidate family containing both registered safety policies
- persistent tug/channel/transfer exclusion over complete transition intervals
- duration-normalized robust scoring with score-semantic dominance pruning
- a separate terminal-cost Pareto audit that cannot override the robust oracle
- separate BACASP source-core and synthetic migration certificates

Not supported:
- deployment in an operating industrial terminal
- safety certification of physical tug, channel, crane, or yard operations
- global optimality for arbitrary berth-allocation or crane-scheduling instances
- using synthetic migration evidence as BACASP source evidence or vice versa
