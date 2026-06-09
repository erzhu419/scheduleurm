# Module51 Production Coverage Drilldown

Date: 2026-06-08

Module49 is a raw-history capacity attempt.  Module51 tightens the
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

## Current Result After Module64

For the 30-day completed/active production window:

```text
completed_active_production records = 2553
representative mapped = 1503
strict mapped = 681
unmapped / measurement-required = 1050
representative mapped_fraction = 0.588719
capacity slack on mapped raw-window representative load from Module49 delta = 0.001497772
global theorem closed = false
```

The newly measured Module56 through Module64 sub-buckets appear as
strict mapped coverage:

```text
freqduet_cpu_ablation_c17_32 = 136 / 2553 completed-active production records
bamor_cpu_training_c3_8_completed_history = 122 / 2553 completed-active production records
freqduet_cpu_ablation_c9_16 = 110 / 2553 completed-active production records
freqduet_runner_v3_c_le2_completed_history = 84 / 2553 completed-active production records
transit_native_promotion_c17_32_seedrange_completed_history = 71 / 2553 completed-active production records
freqduet_cpu_ablation_c33_64_completed_history = 63 / 2553 completed-active production records
sumo_eval_simple_sac_c_le2 = 54 / 2553 completed-active production records
freqduet_cpu_ablation_c3_8_completed_history = 40 / 2553 completed-active production records
freqduet_runner_v3_allfreq_alllayers_c9_16 = 1 / 2553 completed-active production records
```

Module51 now intentionally reports coverage only.  The mapped representative
raw-window load still has positive capacity slack in Module49, but that does
not close the global theorem because measured coverage is still incomplete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `cpu_sumo_transit_eval_or_control` | measurement required | 704 | 0.275754 |
| `generic_cpu_python` | measurement required | 150 | 0.058754 |
| `cpu_eval_generic` | measurement required | 81 | 0.031727 |
| `artifact_io_control` | measurement required | 77 | 0.030161 |
| `scheduler_control_plane` | measurement required | 22 | 0.008617 |
| `gpu_rl_unmeasured_variant` | measurement required | 16 | 0.006267 |

The dominant remaining production-coverage blocker is still the
CPU/SUMO/transit evaluation/control family, but it has been reduced from the
pre-Module56 count:

```text
before Module56: cpu_sumo_transit_eval_or_control = 1245 / 2504
after Module56:  cpu_sumo_transit_eval_or_control = 1208 / 2442
after Module57:  cpu_sumo_transit_eval_or_control = 1211 / 2449
after Module58:  cpu_sumo_transit_eval_or_control = 1103 / 2453
after Module59:  cpu_sumo_transit_eval_or_control = 1049 / 2458
after Module60:  cpu_sumo_transit_eval_or_control = 1009 / 2465
after Module61:  cpu_sumo_transit_eval_or_control = 946 / 2469
after Module62:  cpu_sumo_transit_eval_or_control = 862 / 2471
after Module63:  cpu_sumo_transit_eval_or_control = 803 / 2487
after Module64:  cpu_sumo_transit_eval_or_control = 704 / 2553
```

## Interpretation

This is not a reason to weaken the math.  It identifies the next empirical
closure target:

```text
measure theorem-grade service curves for the remaining cpu_sumo_transit_eval_or_control sub-buckets
then add those buckets to the production load certificate and capacity action slice
```

Until the remaining service buckets are measured or otherwise certified, a
reviewer-facing global production theorem remains open.  The current valid claim
is narrower: the measured mapped representative production load is inside the
current measured service slice, the exact `run_freqduet_ablation.py` c17_32
production sub-slice is strictly measured, and the exact
`runner_v3.py --config configs_freqduet/F_allfreq_alllayers_hiro.yaml` c9_16
sub-slice is strictly measured.  Module58 additionally maps 110 completed-active
c9_16 `run_freqduet_ablation.py` records with parsed per-record units.  Module59
maps 54 clean SimpleSAC c_le2 eval records using a profile-1 completed-history
lower-service certificate.  Module60 maps 40 c3_8 `run_freqduet_ablation.py`
records using a profile-1 completed-history lower-service certificate.  Module61
maps 63 c33_64 `run_freqduet_ablation.py` records the same way.  Module62 maps
84 c_le2 direct `runner_v3.py` records.  Module63 maps 71 completed-active
Transit native-promotion c17_32 seed-range validation records using parsed
seed-episode units.  Module64 maps 122 completed-active BAMOR c3_8 CPU-training
records using parsed training-step units.  The next closure targets are now the
remaining CPU/SUMO/transit residual buckets, led by `sumo_eval_cpu|c_le2`,
`freqduet_cpu_ablation|c_3_8`, and the remaining FreqDuet/Transit residual
command shapes in the regenerated Module53 manifest.
