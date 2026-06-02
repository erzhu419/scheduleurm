# Module 6 Validation: Service Curve Size 8192 Case

Date: 2026-06-03

## Scope

This module extends the fixed-profile service-curve validation from
`size=12288` to a second synthetic workload size, `size=8192`.  It also fixes a
runner bug exposed by this faster workload: if all tasks finish before the end
of the measurement window, the final `done` snapshot has zero running rate, but
the service curve should keep the last valid progress-rate sample.

## Live Run

Run id:

```text
module6_service_curve_jtl110gpu2_size8192_20260603_001
```

Run directory:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module6_service_curve_jtl110gpu2_size8192_20260603_001
```

Command:

```bash
python3 -m algorithm.experiments.service_curve_validation \
  --run-id module6_service_curve_jtl110gpu2_size8192_20260603_001 \
  --node jtl110gpu2 \
  --gpus 0,1 \
  --profiles 1,2,3 \
  --steps 2400 \
  --size 8192 \
  --measure-s 90 \
  --poll-s 30 \
  --warmup-timeout-s 300 \
  --proxy-job-counts 6,12,24 \
  --min-two-vs-three-gain 1.05
```

## Result

Measured service curve:

| Count/GPU | Running | With rate | Mean step/s | Aggregate step/s | Evictions | Blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 34.484897 | 68.969794 | 0 | 0 |
| 2 | 4 | 4 | 16.117634 | 64.470537 | 0 | 0 |
| 3 | 6 | 6 | 10.801971 | 64.811828 | 0 | 0 |

Completion proxy for all `n` jobs finishing:

| n | Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|---:|
| 6 | 1 | 139.191 | 208.787 |
| 6 | 2 | 198.540 | 297.810 |
| 6 | 3 | 222.182 | 222.182 |
| 12 | 1 | 243.585 | 417.574 |
| 12 | 2 | 297.810 | 446.716 |
| 12 | 3 | 333.273 | 444.363 |
| 24 | 1 | 452.372 | 835.148 |
| 24 | 2 | 521.168 | 893.431 |
| 24 | 3 | 555.454 | 888.727 |

Validation verdict:

- all profiles launched on the requested GPUs;
- all profiles produced progress-rate samples;
- `1/GPU` completed before the measurement window ended, and the corrected
  runner retained the last valid running-rate sample;
- no scheduler block was recorded;
- no eviction was recorded;
- `2/GPU` mean per-task service gain over `3/GPU` was `1.492101187`;
- measured best mean-flow and makespan profile for `n=6,12,24` was `1/GPU`;
- final verdict: `PASS`.

## Interpretation

This second size does not prove a universal sweetspot, but it makes the first
result less likely to be a one-size artifact.  For both `size=8192` and
`size=12288`, aggregate throughput at `1/GPU` was higher than at `2/GPU` or
`3/GPU`, so more co-location did not improve total completion time for these
synthetic matmul tasks.

This strengthens the case for adaptive per-class/per-regime service-curve
measurement.  A fixed `sweet_spot_tasks_per_gpu=2` is too rigid; the scheduler
should eventually choose admission/co-location from measured service curves and
the active objective.

LLM-assisted evaluation remains useful only as an advisor for experiment design
and anomaly review.  It cannot replace the logged progress-rate evidence used
for theorem-facing service certificates.
