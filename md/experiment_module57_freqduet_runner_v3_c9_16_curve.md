# Module57 FreqDuet Runner V3 Exact-Config CPU Curve

Date: 2026-06-08

Module57 measures a narrow direct-runner slice inside the remaining
`freqduet_cpu_ablation|c_9_16` production blocker.

## Scope

```text
sub_bucket = freqduet_runner_v3|c_9_16
workload_key = freqduet_runner_v3_allfreq_alllayers_c9_16
node = jtl110cpu2
resource_kind = cpu_sumo_transit
vram_mb/task = 0
cpu/task = 14
ram/task = 32768
progress_unit = episode
progress_units/task = 3
production command shape = runner_v3.py --config configs_freqduet/F_allfreq_alllayers_hiro.yaml
```

This is an exact-config certificate.  It does not certify other
`runner_v3.py` configs in the same c9_16 CPU bucket and does not certify the
larger `run_freqduet_ablation.py` c9_16 family.

## Artifacts

```text
md/experiment_artifacts/module57_freqduet_runner_v3_c9_16_jtl110cpu2_curve_p1248.json
md/experiment_artifacts/module57_freqduet_runner_v3_c9_16_jtl110cpu2_curve_p1248.md
md/experiment_artifacts/module57_freqduet_runner_v3_c9_16_jtl110cpu2_curve_p1248_reports/
```

## Measured Curve

Each task ran `runner_v3.py` with diagnostics CSV polling.  All requested
profiles were feasible on `jtl110cpu2`.

| Profile | Feasible | Running | Blocked | Aggregate episode/s | Mean episode/s |
|---:|:---|---:|---:|---:|---:|
| 1 | yes | 1 | 0 | 0.049480 | 0.049480 |
| 2 | yes | 2 | 0 | 0.092721 | 0.046360 |
| 4 | yes | 4 | 0 | 0.172209 | 0.043052 |
| 8 | yes | 8 | 0 | 0.362757 | 0.045345 |

The service cache loads all four profiles as feasible measured service points.

## Production Closure Effect

The 30-day completed/active production view currently contains one exact match:

```text
freqduet_runner_v3_allfreq_alllayers_c9_16 mapped = 1 / 2449
completed_active_production strict mapped count = 126
completed_active_production representative mapped count = 916
measurement_required = 1533
remaining cpu_sumo_transit_eval_or_control = 1211
```

The narrow slice is theorem-grade, but its coverage effect is intentionally
small.  The next production closure target remains the broader
`run_freqduet_ablation.py` c9_16 family.
