# jtl311linux CNN profile capacity decision

## Observed failure

Empty-state calibration wave 7 was launched only after a clean preflight on
both RTX 2080 8 GB GPUs. Five of the six ResNet-50 AMP training children at
profile 3 naturally completed. The sixth child failed during warmup with
`torch.OutOfMemoryError`: only 3.44 MiB remained when PyTorch requested an
additional 20 MiB. The completed campaign row is therefore a terminal,
non-ready `CAPACITY_BOUNDARY`, not a positive-service observation.

Evidence:

- Source campaign artifact: `critical_gpu_completion_v8_jtl311linux_r07_20260803.json`
- Preserved artifact: `critical_gpu_completion_v8_jtl311linux_r07_unsafe_cnn_p3_20260830.json`
- Preserved failed-child log: `critical_gpu_completion_v8_jtl311linux_r07_cnn_p3_oom_gpu1_2.log`
- Campaign result: 8 ready cells, 1 capacity boundary, 0 wait cells.

## Statistical consequence

The stochastic lower-confidence-bound gate freezes positive candidate support
from training waves 1--3. A first capacity failure in calibration wave 7 is
intentionally fatal: a later successful rerun cannot erase the observed unsafe
action. Continuing waves 8--13 with the same candidate support cannot produce a
valid certificate.

## Action-space decision

For the `gpu_2080_8gb_dual` hardware class, replace the CNN action profiles
`(1, 3)` with a qualification family `(1, 2)`. Other hardware classes retain
their registered CNN profiles. The p2 action first passed an excluded smoke
wave and training waves 1--2, but failed in training wave 3: one child observed
another CNN process at 6.37 GiB, used 1.21 GiB itself, and failed an additional
26 MiB allocation with only 19.5 MiB free. The source log is preserved as
`critical_gpu_completion_v8_jtl311linux_r03_cnn_p2_oom_gpu1_1.log`.

Because this failure occurred inside the pre-registered training split, the
stochastic gate freezes the positive CNN support at p1. P2 remains in the
declared qualification family so every later row is audited, but all later p2
successes are diagnostic only and cannot re-admit the action. Both p2 and p3
therefore have zero lower service in the theorem-facing cache for this hardware
class. The 13-wave campaign continues under the unchanged measurement-code
manifest; it does not restart or discard the observed failure.
