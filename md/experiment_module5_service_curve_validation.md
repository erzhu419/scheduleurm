# Module 5 Validation: Fixed-Profile Service Curve

Date: 2026-06-03

## Scope

This module validates a fixed co-location service-curve runner.  It does not
replace scheduler policy and does not modify legacy placement.  It pins short
progress-bearing benchmark tasks onto selected GPUs, measures stable progress
rates, cancels the tasks, and reports the measured best profile.

The module is intended to support the proof/experiment route:

- `service_units` are real monotone `Step i/N` progress units;
- each fixed profile is a statewise action/regime sample;
- the resulting curve gives empirical inputs for service lower bounds,
  candidate cover calibration, and adaptive sweetspot selection.

## Live Run

Run id:

```text
module5_service_curve_jtl110gpu2_20260603_001
```

Run directory:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module5_service_curve_jtl110gpu2_20260603_001
```

Command:

```bash
python3 -m algorithm.experiments.service_curve_validation \
  --run-id module5_service_curve_jtl110gpu2_20260603_001 \
  --node jtl110gpu2 \
  --gpus 0,1 \
  --profiles 1,2,3 \
  --steps 2400 \
  --size 12288 \
  --measure-s 90 \
  --poll-s 30 \
  --warmup-timeout-s 300 \
  --min-two-vs-three-gain 1.05
```

The first version of the runner incorrectly treated `2/GPU` as a fixed
expected sweetspot.  The measurements themselves were valid, but that fixed
expectation failed because this synthetic matmul workload measured `1/GPU` as
the best 6-job flow-time proxy.  The runner now defaults to reporting the
measured best profile; a fixed expectation is checked only when
`--expected-sweetspot-count` is explicitly set above zero.

## Result

Measured service curve:

| Count/GPU | Running | With rate | Mean step/s | Aggregate step/s | Evictions | Blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 10.269267 | 20.538535 | 0 | 0 |
| 2 | 4 | 4 | 4.784535 | 19.138140 | 0 | 0 |
| 3 | 6 | 6 | 3.226796 | 19.360778 | 0 | 0 |

Completion proxy for all `n` jobs finishing:

| n | Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|---:|
| 6 | 1 | 467.414 | 701.121 |
| 6 | 2 | 668.822 | 1003.232 |
| 6 | 3 | 743.772 | 743.772 |
| 12 | 1 | 817.975 | 1402.242 |
| 12 | 2 | 1003.232 | 1504.848 |
| 12 | 3 | 1115.658 | 1487.544 |
| 24 | 1 | 1519.096 | 2804.484 |
| 24 | 2 | 1755.657 | 3009.697 |
| 24 | 3 | 1859.429 | 2975.087 |

Validation verdict:

- all profiles launched on the requested GPUs;
- all running benchmark tasks produced progress rates;
- no scheduler block was recorded;
- no eviction was recorded;
- `2/GPU` mean per-task service gain over `3/GPU` was `1.482750848`;
- measured best mean-flow and makespan profile for `n=6,12,24` was `1/GPU`;
- final verdict after removing the incorrect fixed `2/GPU` expectation: `PASS`.

## Interpretation

This result does not say the global Scheduleurm sweetspot is always `1/GPU`.
It says the sweetspot is workload/regime dependent, which supports the
adaptive-sweetspot route.  For the current synthetic JAX matmul benchmark on
`jtl110gpu2`, adding co-located tasks reduced per-task service enough that
`1/GPU` dominated the 6-job flow-time proxy, while `2/GPU` still substantially
improved per-task service relative to `3/GPU`.

The next scheduling module should therefore use measured service curves as
input rather than hard-coding `sweet_spot_tasks_per_gpu=2` for every class and
regime.

## Limits and Next Measurements

This is still a single synthetic workload (`size=12288`) on one node.  It is
not enough for a paper claim that `1/GPU` is globally optimal.  The next live
service-curve passes should vary:

- workload size / memory pressure, for example `8192`, `12288`, `16384`;
- workload class, especially real RL training logs rather than only synthetic
  matrix multiplication;
- repeated runs and randomized launch order, so startup skew and cache/warmup
  effects do not dominate;
- larger completion proxy counts, already supported by `--proxy-job-counts`;
- ETA error as a separate diagnostic, not as the primary service certificate.

An LLM API can be useful as an experiment advisor: selecting the next workload
profiles, explaining anomalous curves, and checking reviewer-facing language.
It must not be used as theorem-grade evidence.  The proof-facing constants must
still come from logged progress, calibrated service lower bounds, and capacity
slack calculations.
