# Critical GPU Loaded-State Stochastic LCB Gate

- Status: `FAIL_VALIDATION`
- Node: `node007`
- Ready observations: `18` / `52`

| Scenario | Resident -> target | Upper target s | Lower resident | Lower target | Holdout covered |
|---|---|---:|---:|---:|:---:|

This is a hardware-local certificate for four registered, direction-sensitive two-workload GPU co-location trajectories. Target completion includes initialization, the canonical outer-loop work, checkpoints/final save, and natural exit. Resident service is the counter increment strictly inside the target start/end markers divided by the full overlap duration. One wave-maximum split-conformal margin jointly covers both functionals for all four actions. Every row also hash-verifies the pre-launch nvidia-smi snapshot and rejects an assigned GPU already carrying undeclared load. It neither populates the legacy workload/profile index nor extrapolates to unmeasured mixtures, hardware, or future workloads.
