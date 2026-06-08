# Module58 FreqDuet CPU Ablation c9_16 Curve

Date: 2026-06-08

Module58 measures the next high-impact production CPU/SUMO/transit slice after
Module56 and Module57.

## Scope

```text
sub_bucket = freqduet_cpu_ablation|c_9_16
workload_key = freqduet_cpu_ablation_c9_16
node = jtl110cpu2
resource_kind = cpu_sumo_transit
vram_mb/task = 0
cpu/task = 13
ram/task = 32768
progress_unit = episode
production command shape = scripts/run_freqduet_ablation.py
strict unit rule = parsed jobs times episodes
```

The classifier maps only c9_16 `run_freqduet_ablation.py` records whose work
size is parseable from `--job-start/--job-end` or literal configs and seeds.
Unknown-size commands remain unmeasured.

## Artifacts

```text
md/experiment_artifacts/module58_freqduet_ablation_c9_16_jtl110cpu2_curve_p1248.json
md/experiment_artifacts/module58_freqduet_ablation_c9_16_jtl110cpu2_curve_p1248.md
md/experiment_artifacts/module58_freqduet_ablation_c9_16_jtl110cpu2_curve_p1248_reports/
md/experiment_artifacts/module58_freqduet_ablation_c9_16_completed_wallclock_audit.json
md/experiment_artifacts/module58_freqduet_ablation_c9_16_completed_wallclock_audit.md
```

## Measured Curve

Each controlled task ran a production-like 13-job shard over four FreqDuet
configs and 20 seeds, using `--workers 13 --worker-threads 1` and diagnostics
CSV polling.

| Profile | Feasible | Running | Blocked | Aggregate episode/s | Mean episode/s |
|---:|:---|---:|---:|---:|---:|
| 1 | yes | 1 | 0 | 0.241604 | 0.241604 |
| 2 | yes | 2 | 0 | 0.615229 | 0.307614 |
| 4 | yes | 4 | 0 | 0.755043 | 0.188761 |
| 8 | yes | 8 | 0 | 1.240651 | 0.155081 |

All four profiles are feasible service points on `jtl110cpu2`.

## Completed Wall-Clock Audit

The matched production records are already completed, so Module58 also audits
their realized wall-clock service using parsed command units.

```text
completed-active mapped count = 110
raw-history mapped count = 129
parsed completed-active work = 138900 episode units
min realized completed-active rate = 0.289775 episode/s
median realized completed-active rate = 0.487672 episode/s
```

This audit is not a substitute for the controlled profile curve.  It is a
sanity certificate that the mapped production records have positive parseable
work units and completed with nonzero realized service.

## Production Closure Effect

After Module58:

```text
completed_active_production records = 2453
strict mapped = 236
representative mapped = 1026
measurement_required = 1427
remaining cpu_sumo_transit_eval_or_control = 1103
```

The remaining `freqduet_cpu_ablation|c_9_16` bucket has 50 records with other
command shapes, mostly direct runner or FreqHRL/Transit variants.  The next
Module53 probe target is now `sumo_eval_cpu|c_le2`.
