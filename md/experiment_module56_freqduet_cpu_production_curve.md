# Module56 FreqDuet CPU Production Curve

Date: 2026-06-08

Module56 runs the first theorem-grade service-curve slice for the dominant
`cpu_sumo_transit_eval_or_control` production blocker identified by Module51
and decomposed by Module53.

## Scope

```text
sub_bucket = freqduet_cpu_ablation|c_17_32
workload_key = freqduet_cpu_ablation_c17_32
node = jtl110cpu2
resource_kind = cpu_sumo_transit
vram_mb/task = 0
cpu/task = 24
work_items/task = 24 seeds/workers
progress_unit = episode
progress_units/task = 72
```

This is a measured production sub-slice, not a generic CPU benchmark.  It
strictly covers records whose command invokes `run_freqduet_ablation.py`, whose
GPU request is zero, and whose CPU request lies in the `c_17_32` bucket.  Other
FreqDuet/FreqHRL/TransitDuet records in the same CPU bucket remain unmeasured
unless their command path matches this runner.

## Artifacts

```text
algorithm/experiments/production_cpu_workload_curve.py
algorithm/experiments/csv_progress_wrapper.py
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124.json
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124.md
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124_reports/
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8.json
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8.md
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8_reports/
```

## Measured Curve

Each Scheduleurm task ran 24 seeds/workers and reported progress through
diagnostics CSV polling.  The service cache uses the median measured aggregate
rate when a measurement window has multiple samples.

| Profile | Feasible | Running | Blocked | Aggregate episode/s | Mean episode/s |
|---:|:---|---:|---:|---:|---:|
| 1 | yes | 1 | 0 | 0.447090 | 0.447090 |
| 2 | yes | 2 | 0 | 0.539602 | 0.269801 |
| 4 | yes | 4 | 0 | 0.878374 | 0.219593 |
| 8 | boundary | 5 | 3 | 1.369099 observed on active tasks | 0.273820 observed on active tasks |

Profile 8 is not a feasible service point: three tasks were queued/blocked by
the CPU fit rule while five progressed.  It is therefore loaded into
`ServiceRateCache` only as a capacity boundary.  The feasible measured domain
for this node bucket is `{1, 2, 4}`.

## Production Closure Effect

After adding `freqduet_cpu_ablation_c17_32` to the strict classifier and default
service cache, the reviewer-facing completed/active production view changed to:

```text
completed_active_production records = 2442
mapped = 915
measurement_required = 1527
mapped_fraction = 0.374693
freqduet_cpu_ablation_c17_32 mapped = 125
remaining cpu_sumo_transit_eval_or_control = 1208
```

The mapped production load remains inside the measured capacity slice:

```text
strict raw-history mapped delta = 0.319917853
representative raw-history mapped delta = 0.272849952
completed-active representative delta = 0.311461063
```

This closes the exact `run_freqduet_ablation.py` c17_32 production sub-slice. It
does not close the whole original `freqduet_cpu_ablation|c_17_32` group, because
116 completed/active records in that group use different command shapes such as
native validation or direct runner invocations.  It also does not close global
production stability, because the remaining `cpu_sumo_transit_eval_or_control`
sub-buckets and several smaller CPU/GPU production buckets still require their
own measured service curves or theorem-grade equivalence certificates.
