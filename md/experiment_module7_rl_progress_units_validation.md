# Module 7: RL Progress-Unit Validation

This module closes a measurement gap exposed by the first fixed-profile service
curve runs.  The JAX matmul benchmark is a compute-bound synthetic workload, so
its measured curve favored 1/GPU.  RL training jobs can be different: env
stepping, replay sampling, Python/JAX dispatch, evaluation, and checkpointing
create GPU-idle gaps.  When each task uses the GPU in short bursts, 4-5 tasks on
one GPU can fill those gaps and keep each task's wall time close to the solo
case.  In that regime, the correct service curve can be nearly flat in per-task
rate through 4/GPU or 5/GPU, so aggregate service increases with packing.

The scheduler must therefore measure task-native progress by workload class,
not reuse the synthetic matmul sweet spot as a global rule.

## Change

Added `algorithm/experiments/progress_units.py`.

It parses service-rate signals including:

- `Step 124/2400 ... rate=10.353019 step/s`
- `Iter 1989 | ... | 10.6s/iter`
- current/total fallbacks from scheduler runtime fields and command flags such
  as `--max_iters 2000`

`algorithm/experiments/sweetspot_ab_validation.py` now records both generic
`rate_unit_s` fields and the old compatibility aliases.  The service-curve
verdict now records `rate_units` and fails a profile if measured tasks mix
incompatible units.

## Validation

Targeted checks passed:

```text
PASS progress parser reads benchmark current/total
PASS progress parser reads benchmark step/s rate
PASS progress parser reads RL iter and cmd total
PASS progress parser converts s/iter to iter/s
PASS progress parser uses latest RL progress line
PASS task progress parser fills scheduler runtime counters
PASS sweetspot summary counts RL iter/s rates
PASS service curve keeps last valid rate sample after short jobs finish
PASS service curve verdict reports one-per-GPU measured best without failing fixed expectation
PASS service curve verdict fails incomplete/blocked profiles
PASS service curve verdict passes two-per-GPU flow sweetspot
```

Compile checks passed for the touched modules.

Full `skill/test_regression.py` is not clean in the current baseline.  The run
still hits existing requeue/SlurmBackend/staging failures unrelated to this
progress parser module, plus the environment lacks a plain `python` binary
unless a temporary `python -> python3` shim is added.  I did not count the full
suite as this module's pass condition.

## Interpretation

This change does not claim that 5/GPU is always optimal.  It makes 5/GPU
measurable for RL classes.  The next controlled validation should run the same
RL task class under fixed profiles such as 1/2/3/4/5 per GPU and compare
completion-time proxy and aggregate `iter/s` using these parser outputs.
