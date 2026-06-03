# Module 9: RE-SAC Dense Service-Curve Validation

Date: 2026-06-03

Run id: `module11_resac_walker_dense_curve_jtl110gpu2_gpu1_20260603_001`

Run directory: `/home/erzhu419/.claude/scheduler/experiments/runs/module11_resac_walker_dense_curve_jtl110gpu2_gpu1_20260603_001`

Workload: RE-SAC `sac` on `Walker2d-v2`, `max_iters=60`, `ensemble_size=2`, one pinned `jtl110gpu2:GPU1` profile at a time.

## Validation Scope

This run is a dense scheduler/harness validation, not a replacement for the paper direction in `math.md`. It validates that the experimental action surface can represent and execute concrete co-location actions of the form "place this task family on this GPU with count c", and it gives a first service-curve observation for the proof-side capacity and drift calibration pipeline.

The user's real RL workload may have a flatter 4-5/GPU curve than this smoke template. That is not contradicted here; it means the same dense runner must be used again with the exact production training command/template before claiming production sweet spots.

## Code Fixes From This Run

- Fixed `--require-gpu` hard-pin semantics in `skill/scheduler.py`: LocalBackend claim-race retry no longer falls back to another GPU when `require_gpu_idx` is set.
- Added `skill/tests/test_gpu_hard_pin_claim_retry.py` to cover the hard-pin claim retry case.
- Added strict actual-placement validation to service-curve verdict rows. Cross-GPU contamination like `{"0": 1, "1": 14}` is now invalid and excluded from best-profile selection.
- Added capacity-boundary reporting so a blocked saturation profile can be recorded as a boundary instead of forcing manual timeout handling.

## Dense Curve

All valid profiles from 1/GPU through 14/GPU were placed on `jtl110gpu2:GPU1` only. The 15/GPU profile reached the capacity boundary: 14 tasks ran on GPU1 and the 15th stayed queued with:

`CLAIM_RACE: gpu1: gpu1: post-claim free 288MB < margin 500MB`

| Count/GPU | Running | With rate | Placement | Boundary | Mean iter/s | Aggregate iter/s | Min iter/s | Max iter/s |
|---:|---:|---:|:---|:---|---:|---:|---:|---:|
| 1 | 1 | 1 | ok | no | 0.588235 | 0.588235 | 0.588235 | 0.588235 |
| 2 | 2 | 2 | ok | no | 0.294118 | 0.588235 | 0.294118 | 0.294118 |
| 3 | 3 | 3 | ok | no | 0.024132 | 0.072396 | 0.020121 | 0.028409 |
| 4 | 4 | 4 | ok | no | 0.018701 | 0.074806 | 0.018553 | 0.018904 |
| 5 | 5 | 5 | ok | no | 0.017468 | 0.087341 | 0.016892 | 0.018149 |
| 6 | 6 | 6 | ok | no | 0.093040 | 0.558241 | 0.015699 | 0.476190 |
| 7 | 7 | 7 | ok | no | 0.016095 | 0.112664 | 0.015385 | 0.017301 |
| 8 | 8 | 8 | ok | no | 0.066844 | 0.534749 | 0.013038 | 0.196078 |
| 9 | 9 | 9 | ok | no | 0.064563 | 0.581071 | 0.014859 | 0.108696 |
| 10 | 10 | 10 | ok | no | 0.052007 | 0.520068 | 0.007758 | 0.128205 |
| 11 | 11 | 11 | ok | no | 0.049247 | 0.541715 | 0.010593 | 0.106383 |
| 12 | 12 | 12 | ok | no | 0.042418 | 0.509019 | 0.011390 | 0.357143 |
| 13 | 13 | 13 | ok | no | 0.043343 | 0.563453 | 0.010695 | 0.090909 |
| 14 | 14 | 14 | ok | no | 0.037310 | 0.522346 | 0.008643 | 0.121951 |
| 15 | 14 | 14 | invalid | yes | 0.035979 | 0.503708 | 0.008691 | 0.121951 |

## Interpretation

The strict capacity boundary for this template on `jtl110gpu2:GPU1` is 14 concurrent tasks under the current 800MB estimate and 500MB claim margin. The 15th task does not spill to GPU0 after the hard-pin fix; it stays queued, which is the correct semantics for a single-GPU service-curve experiment.

The curve is noisy because the measurement window is short and the RL command has slow first-iteration/evaluation behavior. Several counts have a high max-rate outlier that lifts aggregate throughput while the tail remains slow. For paper calibration, use lower-bound or robust statistics from repeated runs, not only the last-window aggregate.

For this smoke workload, the completion proxy selects 1/GPU as best for `n=15,30,60`. That should not be generalized to the user's production RL jobs until the exact production task template is run through the same dense protocol.

## Verification

- `python3 -m py_compile algorithm/experiments/service_curve_validation.py algorithm/experiments/workload_service_curve_validation.py skill/scheduler.py`
- Targeted regression loader over:
  - `skill/tests/test_gpu_hard_pin_claim_retry.py`
  - `skill/tests/test_workload_service_curve_validation.py`
  - `skill/tests/test_service_curve_validation.py`
- Result: 14 targeted checks passed, 0 failed.

## Next Experiment Hook

The next run should replace the smoke RE-SAC command with the user's exact RL training command where 4-5/GPU is believed to be near-flat. Keep the dense profile list and strict placement validation. Then feed the repeated robust curve into the `math.md` calibration path for `L`, `rho`, `epsilon_est`, service lower bounds, and drift slack.
