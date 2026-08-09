# MMRCPSP v1 holdout repair regression

- Status: `PASS`
- Instances: `56/56` feasible after the general renewable-completability fix.
- Ours Pareto-nondominated: `56/56` against the four registered priority rules.
- Prospective external holdout claim ready: `false`.
- Reason: all instances were opened by the v1 holdout before the repair.
- Original v1 result remains a fail-closed prospective failure and is not overwritten.
- Repair commit: `c62cddd198d15faaa3d9c2d36415142a42caf727`.

This artifact is a development regression gate. A disjoint post-fix PSPLIB holdout is required for prospective evidence.
