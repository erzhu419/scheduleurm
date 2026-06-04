# RE-SAC Progress Wrapper ETA Validation

Date: 2026-06-04

This note records the instrumentation fix for real RL workload measurements:
the dispatched task now emits task-native progress lines that the scheduler can
parse directly.  The purpose is to avoid estimating ETA only from scheduler
elapsed time during JAX warmup/compile, where early projections can be wildly
wrong.

## Change

- Added `algorithm/experiments/progress_wrapper.py`.
- Updated the RE-SAC and BAPR real-workload command templates to execute
  through the wrapper.
- Updated `workload_service_curve_validation` to deploy the wrapper and
  `progress_units.py` to the target node under
  `/tmp/scheduleurm_progress_wrapper_pkg` before submitting tasks.
- The wrapper preserves the original training log and appends normalized lines:

```text
ScheduleurmProgress Iter 2/1500 rate=0.0331125828 iter/s ETA 45239.6s seconds_per_iter=30.2 source=seconds_per_unit
```

This is line-oriented rather than a terminal carriage-return tqdm bar, because
Scheduleurm tails remote logs line by line.

## Validation Run

- Run id:
  `module19_resac_progress_wrapper_smoke_1gpu_jtl110gpu2_gpu1_20260604_001`
- Node/GPU: `jtl110gpu2:GPU1`
- Workload: RE-SAC Ant real workload template
- Task id: `t6687`
- Measurement mode: short warmup plus short measurement window, then cancel.
  The task was not run to completion.

Observed scheduler status while running:

```text
eta_source=inline_eta
eta_confidence=high
runtime_current_unit=1
runtime_total_units=1500
last_progress_line=ScheduleurmProgress Iter 1/1500 rate=0.035971223 iter/s ETA 41672.2s seconds_per_iter=27.8 source=seconds_per_unit
```

The runner verdict passed:

- `pass=true`
- `running_with_rate_count=1`
- `rate_units=["iter"]`
- median aggregate rate for the short measurement window:
  `0.0345419029 iter/s`

## Parser Bug Fixed

The first wrapper version incorrectly treated banner metadata such as
`Updates/iter: 250  Samples/iter: 4000` as a progress counter.  The parser now:

- rejects labels preceded by `/`, so unit metadata like `/iter` is ignored;
- rejects `current > total` after command/template total inference;
- suppresses impossible wrapper output such as `Iter 4000/1500`.

Post-fix local and remote wrapper smoke both preserve the banner without
emitting fake progress, then emit progress only for real `Iter` training lines.

## Tests

- `python3 -m py_compile ...` passed for the changed experiment modules.
- Targeted test loader passed: `checks=38 failed=0`.
- Remote wrapper smoke on `jtl110gpu2` passed after deploying the updated wrapper
  to `/tmp/scheduleurm_progress_wrapper_pkg`.

## Implication

Future dense service-curve experiments can stop after a stable progress window:
they do not need to wait for every RL training task to finish.  ETA and
throughput measurement now come from task-native progress lines, with the
service-curve runner using median aggregate-rate samples over the measurement
window.
