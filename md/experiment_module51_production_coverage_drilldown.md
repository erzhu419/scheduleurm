# Module51 Production Coverage Drilldown

Date: 2026-06-08

Module49 was a raw-history capacity attempt.  This module tightens the
reviewer-facing population definition before making any global production
stability claim.

## Artifacts

```text
algorithm/experiments/production_coverage_drilldown.py
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module51_production_coverage_drilldown.md
```

## Why This Module Was Needed

The raw 30-day Scheduleurm history contains cancelled tasks, benchmark tasks,
guard tests, and adopted compiler helper processes.  Those records are useful
for operational debugging, but they are not a clean production arrival
population for an OR theorem.

Module51 separates:

```text
raw_history_all
attempted_production
completed_active_production
```

The theorem-facing view should be declared explicitly.  The current most
defensible view is `completed_active_production`: completed plus currently
active production-like tasks, excluding benchmark/test/cancel/forgotten records
and adopted compiler helper processes.

## Current Result

For the 30-day window:

```text
completed_active_production records = 2495
representative mapped = 948
unmapped / measurement-required = 1547
mapped_fraction = 0.379960
capacity slack on mapped representative load delta = 0.306553655
global theorem closed = false
```

The mapped representative load still has positive capacity slack.  That does
not close the global theorem because coverage is far from complete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `cpu_sumo_transit_eval_or_control` | measurement required | 1227 | 0.491784 |
| `generic_cpu_python` | measurement required | 136 | 0.054509 |
| `cpu_eval_generic` | measurement required | 89 | 0.035671 |
| `artifact_io_control` | measurement required | 77 | 0.030862 |
| `gpu_rl_unmeasured_variant` | measurement required | 16 | 0.006413 |
| `scheduler_control_plane` | measurement required | 2 | 0.000802 |

The dominant production-coverage blocker is therefore not q01/q11 GPU RL.  It
is the CPU/SUMO/transit evaluation/control family:

```text
cpu_sumo_transit_eval_or_control = 1227 / 2495 completed-active production records
```

## Interpretation

This is not a reason to weaken the math.  It identifies the next empirical
closure target:

```text
measure theorem-grade service curves for cpu_sumo_transit_eval_or_control
then add that bucket to the production load certificate and capacity action slice
```

Until that service bucket is measured or otherwise certified, a reviewer-facing
global production theorem remains open.  The current valid claim is narrower:
the measured mapped representative production load is inside the current
measured service slice, but the declared global production population is not yet
fully covered.
