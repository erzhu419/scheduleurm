# Final ETA/OR closure status (2026-08-31)

## Closed in the current measurement code

- `jtl311linux` empty-state campaign: 13 frozen waves and the hardware-local stochastic lower-confidence-bound (LCB) gate pass.
- `jtl311linux` loaded-state campaign: 13 frozen waves, 52 ready observations, and the four direction-sensitive two-workload action bounds pass under measurement hash `a410bdae44dad85e3adacfff9bbf2f0ff0a3c9e74c6568cda26abb6643a5ffe1`.
- All-hardware empty-state cache: `jtl110gpu`, `jtl110gpu2`, `node007`, and `jtl311linux` remain separate operational execution classes. The cache/report hash link passes.
- Empty-state normalized slack: all four hardware-local four-class models pass at load fraction `0.8`, with minimum `eta = 0.05`. The displayed four-node simultaneous coverage lower bound is the conservative union bound `0.6`; rates are not pooled across nodes.
- OR generalization v7: FJSP, MMRCPSP, and port protocol gates pass with all registered negative results retained.

## Migration recalibration available before loaded-ledger closure

The empty-state all-hardware cache already contains every exact state used by the four registered controlled migration specifications, including the CPU half/full resident states. Recalculation at 25%, 50%, and 75% progress therefore produces 12/12 exact, cache-bound rows without legacy or state fallback.

All 12 current-direction actions are non-beneficial under

\[
\frac{W}{\mu_{\mathrm{old}}}-\frac{W}{\mu_{\mathrm{new}}}>K.
\]

This is retained as a negative result. It does not imply that migration is never useful; it states only that the four registered source/destination/load-state directions do not amortize their measured checkpoint, synchronization, staging, resume, lost-work, and risk costs at the three registered progress points.

## Remaining hard blocker

The final all-hardware loaded action ledger is not closed. At the 2026-08-31 snapshot, all four `node007` GPUs were occupied by non-campaign CSBAPR processes at 97--99% utilization, so no benchmark was launched and no running task was touched.

The current frozen loaded measurement hash is `a410bdae44dad85e3adacfff9bbf2f0ff0a3c9e74c6568cda26abb6643a5ffe1`. Existing `node007` loaded artifacts use older measurement hashes (`2da...` or `c920...`). A split-conformal certificate cannot combine those waves with new waves under the current code. Consequently, when `node007` is clean, the required work is a complete current-hash r01--r13 loaded campaign, not only r11--r13.

Until that campaign passes, the following remain intentionally blocked:

- the four-node loaded action ledger and final CPU+GPU loaded service cache;
- canonical migration recalibration against that final loaded cache;
- the unified hardware replay, SOTA envelope, ablation, and four-quadrant figure generated from the final cache;
- final manuscript-number synchronization and the submission PDF rebuild.

No direct full-stack SOTA-binary superiority or arbitrary future-workload claim follows from the currently closed artifacts.
