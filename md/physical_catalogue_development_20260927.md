# Four-node complete-frame catalogue development

The existing small hardware cache and loaded-action ledger now build four
node-local complete-frame point models, with three distinct ordered requirement
topologies. No server data were pulled. The original gpu_b snapshot is
unchanged field for field. Sources, phase accounting, and model snapshots are in
[the registered protocol](experiment_artifacts/aligned_hardware_physical_catalogue_screen_development_20260927_protocol.json).

The fixed grid `(0, 1, 3, 4, 10)^3` was evaluated at each node's native lane count.
All 500 full/reduced oracle comparisons preserve the exact MaxWeight score;
all 1,000 K4096 B&B calls agree with their own catalogue's Fraction oracle.

| Node | Lanes | Options before/after | Canonical lift | Action mismatches / 125 |
| --- | ---: | ---: | --- | ---: |
| gpu_b | 2 | 9 / 8 | valid | 0 |
| gpu_a | 2 | 9 / 8 | valid | 0 |
| cpu_a | 2 | 8 / 7 | rejected: dominator has later index | 0 |
| node007 | 4 | 9 / 8 | rejected: dominator has later index | 15 |

For node007, queue `(1, 0, 10)` selects full indices `(0, 8, 8, 8)` versus reduced
indices mapped to `(1, 8, 8, 8)`. Both frames last 24.097317012424657 seconds and
have the same score. The slower, earlier action remains the canonical choice
because another lane sets the frame duration. The known R2 `(1, 0, 4)` case also
reproduces at full precision. No catalogue was reordered to bypass the lift rule.

[Result](experiment_artifacts/aligned_hardware_physical_catalogue_screen_development_20260927.json):
`CANONICAL_RESTORATION_REQUIRED`. Only one requirement topology passes the current
canonical lift, so R8/R16 scaling remains gated. Next implement a canonical
representative for each requirement group at each frame-duration threshold,
including candidate tie keys and branch bounds, then reuse this oracle panel.

Validation: 11 targeted tests passed. Reproduce with
`python3 -m algorithm.experiments.aligned_hardware_physical_catalogue_screen --output /tmp/catalogue.json --register`,
then the same command without `--register`.

## Limitations

These are cached phase point models and a diagnostic development grid, including
a known counterexample. They do not transfer the gpu2 joint95 guarantee or
establish trajectory, flow-time, wall-time, or independent-confirmation gains.
The paper is unchanged.
