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

## Current Result After Module60

For the 30-day completed/active production window:

```text
completed_active_production records = 2465
representative mapped = 1120
strict mapped = 330
unmapped / measurement-required = 1345
representative mapped_fraction = 0.454361
capacity slack on mapped raw-window representative load from Module49 delta = 0.001497772
global theorem closed = false
```

The newly measured Module56, Module57, Module58, Module59, and Module60 sub-buckets appear as
strict mapped coverage:

```text
freqduet_cpu_ablation_c17_32 = 125 / 2465 completed-active production records
freqduet_cpu_ablation_c9_16 = 110 / 2465 completed-active production records
sumo_eval_simple_sac_c_le2 = 54 / 2465 completed-active production records
freqduet_cpu_ablation_c3_8_completed_history = 40 / 2465 completed-active production records
freqduet_runner_v3_allfreq_alllayers_c9_16 = 1 / 2465 completed-active production records
```

Module51 now intentionally reports coverage only.  The mapped representative
raw-window load still has positive capacity slack in Module49, but that does
not close the global theorem because measured coverage is still incomplete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `cpu_sumo_transit_eval_or_control` | measurement required | 1009 | 0.409331 |
| `generic_cpu_python` | measurement required | 149 | 0.060446 |
| `cpu_eval_generic` | measurement required | 80 | 0.032454 |
| `artifact_io_control` | measurement required | 77 | 0.031237 |
| `gpu_rl_unmeasured_variant` | measurement required | 16 | 0.006491 |
| `scheduler_control_plane` | measurement required | 14 | 0.005680 |

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
records using a profile-1 completed-history lower-service certificate.  The
next closure targets are now the remaining FreqDuet CPU buckets, led by
`freqduet_cpu_ablation|c_33_64` in the regenerated Module53 manifest.
