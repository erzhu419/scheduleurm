# Module51 Production Coverage Drilldown

Date: 2026-06-09

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

## Current Result After Module69

For the 30-day completed/active production window:

```text
completed_active_production records = 2781
representative mapped = 1952
strict mapped = 1041
unmapped / measurement-required = 829
representative mapped_fraction = 0.701906
capacity slack on mapped raw-window representative load from Module49 delta = 0.001497772
global theorem closed = false
```

The newly measured Module56 through Module69 sub-buckets appear as
strict mapped coverage:

```text
freqduet_cpu_ablation_c17_32 = 153 / 2781 completed-active production records
freqduet_cpu_ablation_c9_16 = 110 / 2781 completed-active production records
bamor_mujoco_c3_8_completed_history = 109 / 2781 completed-active production records
transit_native_promotion_c17_32_seedrange_completed_history = 91 / 2781 completed-active production records
freqduet_runner_v3_c3_8_completed_history = 86 / 2781 completed-active production records
freqduet_runner_v3_c_le2_completed_history = 84 / 2781 completed-active production records
freqduet_cpu_ablation_c33_64_completed_history = 63 / 2781 completed-active production records
bamor_train_compare_c3_8_completed_history = 58 / 2781 completed-active production records
transit_native_promotion_c65p_completed_history = 57 / 2781 completed-active production records
sumo_eval_simple_sac_c_le2 = 54 / 2781 completed-active production records
zsw_tsp_sumo_eval_c_le2_completed_history = 50 / 2781 completed-active production records
transit_native_promotion_c33_64_batch_completed_history = 47 / 2781 completed-active production records
freqduet_cpu_ablation_c3_8_completed_history = 40 / 2781 completed-active production records
bamor_diagnostic_shard_c3_8_completed_history = 25 / 2781 completed-active production records
freqduet_cpu_ablation_c65p_completed_history = 7 / 2781 completed-active production records
freqduet_promoted_ep100_c65p_completed_history = 6 / 2781 completed-active production records
freqduet_runner_v3_allfreq_alllayers_c9_16 = 1 / 2781 completed-active production records
```

Module51 now intentionally reports coverage only.  The mapped representative
raw-window load still has positive capacity slack in Module49, but that does
not close the global theorem because measured coverage is still incomplete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `cpu_sumo_transit_eval_or_control` | measurement required | 471 | 0.169364 |
| `generic_cpu_python` | measurement required | 153 | 0.055016 |
| `cpu_eval_generic` | measurement required | 82 | 0.029486 |
| `artifact_io_control` | measurement required | 77 | 0.027688 |
| `scheduler_control_plane` | measurement required | 30 | 0.010787 |
| `gpu_rl_unmeasured_variant` | measurement required | 16 | 0.005753 |

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
after Module65:  cpu_sumo_transit_eval_or_control = 659 / 2589
after Module66:  cpu_sumo_transit_eval_or_control = 576 / 2679
after Module67:  cpu_sumo_transit_eval_or_control = 576 / 2679
after Module68:  cpu_sumo_transit_eval_or_control = 532 / 2755
after Module69:  cpu_sumo_transit_eval_or_control = 471 / 2781
```

Module67 is a capacity-certification refinement rather than a coverage increase:
it splits the broad BAMOR c3_8 class into script-level service classes so the
mapped LP remains theorem-usable under conservative lower-service rates.
Module68 is a coverage increase and a service-refinement step: it closes the
c33_64 native-promotion batch seed-unit class while keeping single-seed smoke
records separate.
Module69 is also a coverage increase: it closes the c65p high-CPU residual by
splitting it into c65p ablation, promoted ep100 shell-batch, and native-promotion
service classes.

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
seed-episode units.  Module64 introduced the broad BAMOR c3_8 CPU-training
completed-history certificate; Module67 refines it into train-compare, Mujoco,
and diagnostic-shard script-level classes.  Module65 maps 50 completed-active
ZSW TSP/SUMO c_le2 runner records using parsed simulated-second units.  Module66
maps 86 c3_8 direct `runner_v3.py` FreqDuet records.  Module68 maps 45
c33_64 native-promotion batch records with parsed seed-episode units.  Module69
maps 57 c65p native-promotion records, 7 c65p `run_freqduet_ablation.py`
records, and 6 c65p promoted ep100 shell-batch records.  The next closure targets
are now the remaining CPU/SUMO/transit residual buckets, led by
`transit_freqhrl_cpu_validation|c_le2`, `freqduet_cpu_ablation|c_9_16`, and
`sumo_eval_cpu|c_le2` in the regenerated Module53 manifest.
