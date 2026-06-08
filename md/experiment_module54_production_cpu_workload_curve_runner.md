# Module54 Production CPU Workload Curve Runner

Date: 2026-06-08

Module53 identified `cpu_sumo_transit_eval_or_control` as the dominant
production coverage blocker.  Module54 adds the concrete CPU-only service-curve
runner for that blocker: it submits real production workload commands through
Scheduleurm, wraps them with progress extraction, waits only until stable
progress is observed, writes profile summaries, and cancels the profile batch.

## Artifacts

```text
algorithm/experiments/production_cpu_workload_curve.py
algorithm/experiments/csv_progress_wrapper.py
md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.json
md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.md
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124.json
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8.json
```

## Runner Features

The runner supports both Linux CPU nodes and the Windows CPU nodes
`jtl110cpu`/`jtl110cpu2`.  For Windows nodes it deploys a small progress-wrapper
package over SSH and runs the child workload in argv mode rather than through
`bash -lc`.

For FreqDuet ablation commands, line-level stdout is not reliable because
`run_freqduet_ablation.py` captures child runner output.  Module54 therefore
adds `csv_progress_wrapper.py`, which polls diagnostics CSV files and emits
standard Scheduleurm progress lines:

```text
ScheduleurmProgress Episode 42/72 rate=... episode/s source=csv_poll
```

The runner also expands each submitted Scheduleurm task into multiple work
items.  For the first production sub-bucket, each task runs 24 seeds/workers;
this avoids the earlier single-seed dry-run mistake and matches the real
`c_17_32` CPU request shape.

## Original Dry-Run Plan

```text
run_id = module54_freqduet_cpu_c17_32_plan
sub_bucket = freqduet_cpu_ablation|c_17_32
node = local
profiles = 1, 2, 4, 8
task_count = 15
cpu/task = 24
ram_mb/task = 65536
total_units/task = 20 episodes
theorem_status = plan_only_not_measured
```

That plan was retained as a reviewable command manifest.  The measured
production run was executed on `jtl110cpu2` with Windows CSV polling and
24 work items per submitted task.  The theorem-grade strict classifier only
maps records that actually invoke `run_freqduet_ablation.py`; other c17_32
records remain in the open manifest.

## Validated Production Run

```text
run_id = module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124
node = jtl110cpu2
sub_bucket = freqduet_cpu_ablation|c_17_32
profiles = 1,2,4
work_items/task = 24
total_units/task = 72 episode rows
verdict = pass
best feasible aggregate profile = 4
```

Measured feasible service curve:

| Profile | Aggregate episode/s | Mean episode/s | Blocks | Evictions |
|---:|---:|---:|---:|---:|
| 1 | 0.447090 | 0.447090 | 0 | 0 |
| 2 | 0.539602 | 0.269801 | 0 | 0 |
| 4 | 0.878374 | 0.219593 | 0 | 0 |

Capacity-boundary run:

```text
run_id = module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8
profile = 8
running/progressing = 5
queued/blocked = 3
capacity_boundary = true
```

Profile 8 is not used as a feasible service point.  It is loaded into
`ServiceRateCache` only as a boundary, so `missing_profiles` and replay policies
do not silently extrapolate to 8-way placement.

## Interpretation

Module54 is no longer dry-run-only.  It now provides the production CPU curve
runner, Windows CSV progress extraction, worker-heavy seed expansion, and
capacity-boundary detection used by Module56.  Module56 closes the first exact
production slice, `run_freqduet_ablation.py` within
`freqduet_cpu_ablation|c_17_32`; the remaining `cpu_sumo_transit_eval_or_control`
sub-buckets, including residual c17_32 command shapes, still need the same style
of progress-bearing measurement before global production stability can be
claimed.
