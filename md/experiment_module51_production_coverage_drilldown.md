# Module51 Production Coverage Drilldown

Date: 2026-06-10

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
active production-like tasks, excluding benchmark/test/cancel/forgotten records,
adopted compiler helper processes, and unobservable external auto-adopted
stdin/wait-for control processes that have no scheduler id, no scheduler log,
and no reproducible progress unit.

## Current Result After Module76

For the 30-day completed/active production window:

```text
completed_active_production records = 2766
representative mapped = 2269
strict mapped = 1361
unmapped / measurement-required = 497
representative mapped_fraction = 0.820318
capacity slack on mapped raw-window representative load from Module49 delta = 0.000050804
global theorem closed = false
```

Module73 is not a service-curve shortcut.  It tightens the theorem-facing
population: external `auto-adopted` stdin or `scheduler.py wait-for` processes
with no scheduler id and no log are not scheduler-controlled arrivals.  They
remain operational telemetry/background-load evidence, but they are outside the
arrival stream that Scheduleurm can assign to service actions.

The newly measured Module56 through Module76 sub-buckets appear as
strict mapped coverage:

```text
freqduet_cpu_ablation_c17_32 = 153 / 2766 completed-active production records
bamor_mujoco_c3_8_completed_history = 152 / 2766 completed-active production records
freqduet_cpu_ablation_c9_16 = 116 / 2766 completed-active production records
transit_native_promotion_c17_32_seedrange_completed_history = 91 / 2766 completed-active production records
freqduet_runner_v3_c3_8_completed_history = 86 / 2766 completed-active production records
freqduet_runner_v3_c_le2_completed_history = 85 / 2766 completed-active production records
freqduet_cpu_ablation_c33_64_completed_history = 63 / 2766 completed-active production records
bamor_train_compare_c3_8_completed_history = 58 / 2766 completed-active production records
transit_native_promotion_c33_64_batch_completed_history = 55 / 2766 completed-active production records
sumo_eval_simple_sac_c_le2 = 54 / 2766 completed-active production records
zsw_tsp_sumo_eval_c_le2_completed_history = 50 / 2766 completed-active production records
transit_native_promotion_c65p_completed_history = 49 / 2766 completed-active production records
freqduet_cpu_ablation_c3_8_completed_history = 40 / 2766 completed-active production records
transit_native_promotion_c9_16_residual_completed_history = 40 / 2766 completed-active production records
bamor_diagnostic_shard_c3_8_completed_history = 25 / 2766 completed-active production records
transit_native_promotion_c_le2_completed_history = 19 / 2766 completed-active production records
transit_trading_policy_c_le2_completed_history = 18 / 2766 completed-active production records
transit_trading_sweep_c_le2_completed_history = 14 / 2766 completed-active production records
freqduet_runner_v3_c9_16_residual_completed_history = 12 / 2766 completed-active production records
cfcmt_policy_rollout_c_le2_completed_history = 10 / 2766 completed-active production records
transit_native_promotion_c9_16_bounded_wait_completed_history = 7 / 2766 completed-active production records
transit_native_control_c_le2_completed_history = 7 / 2766 completed-active production records
freqduet_cpu_ablation_c65p_completed_history = 7 / 2766 completed-active production records
freqduet_promoted_ep100_c65p_completed_history = 6 / 2766 completed-active production records
cfcmt_feed_conversion_c_le2_completed_history = 5 / 2766 completed-active production records
cfcmt_env_validation_c_le2_completed_history = 5 / 2766 completed-active production records
cfcmt_snapshot_generation_c_le2_completed_history = 5 / 2766 completed-active production records
cfcmt_sumo_generation_c_le2_completed_history = 4 / 2766 completed-active production records
transit_surrogate_validation_c_le2_completed_history = 3 / 2766 completed-active production records
cfcmt_traffic_signal_phase2_c_le2_completed_history = 1 / 2766 completed-active production records
transit_freqhrl_import_smoke_c_le2_completed_history = 1 / 2766 completed-active production records
freqduet_runner_v3_allfreq_alllayers_c9_16 = 1 / 2766 completed-active production records
freqduet_runner_v3_c17_32_completed_history = 16 / 2766 completed-active production records
freqduet_paper_longtrain_c17_32_completed_history = 16 / 2766 completed-active production records
transit_native_promotion_c17_32_residual_completed_history = 14 / 2766 completed-active production records
bamor_diagnostic_shard_c9_16_completed_history = 34 / 2766 completed-active production records
bamor_train_compare_c9_16_completed_history = 6 / 2766 completed-active production records
bamor_mujoco_c9_16_completed_history = 3 / 2766 completed-active production records
transit_freqhrl_analysis_matrix_c_le2_completed_history = 11 / 2766 completed-active production records
transit_freqhrl_merge_c_le2_completed_history = 6 / 2766 completed-active production records
freqduet_preflight_c_le2_completed_history = 5 / 2766 completed-active production records
freqduet_baseline_rule_c_le2_completed_history = 5 / 2766 completed-active production records
freqduet_cpu_ablation_c_le2_completed_history = 3 / 2766 completed-active production records
```

Module51 now intentionally reports coverage only.  The mapped representative
raw-window load still has positive capacity slack in Module49, but that does
not close the global theorem because measured coverage is still incomplete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `generic_cpu_python` | measurement required | 143 | 0.051699 |
| `cpu_sumo_transit_eval_or_control` | measurement required | 139 | 0.050253 |
| `cpu_eval_generic` | measurement required | 82 | 0.029646 |
| `artifact_io_control` | measurement required | 77 | 0.027838 |
| `scheduler_control_plane` | measurement required | 40 | 0.014461 |
| `gpu_rl_unmeasured_variant` | measurement required | 16 | 0.005785 |

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
after Module70:  cpu_sumo_transit_eval_or_control = 409 / 2797
after Module71:  cpu_sumo_transit_eval_or_control = 350 / 2805
after Module72:  cpu_sumo_transit_eval_or_control = 320 / 2840
after Module73:  cpu_sumo_transit_eval_or_control = 258 / 2766
after Module74:  cpu_sumo_transit_eval_or_control = 212 / 2766
after Module75:  cpu_sumo_transit_eval_or_control = 169 / 2766
after Module76:  cpu_sumo_transit_eval_or_control = 139 / 2766
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
Module70 closes the former top `transit_freqhrl_cpu_validation|c_le2` blocker
and a few low-CPU native Transit records that the broad keyword manifest had
previously grouped under `freqduet_cpu_ablation|c_le2`.  It splits them into
six strict completed-history service classes rather than merging unlike scripts.
Module71 closes the previous `freqduet_cpu_ablation|c_9_16` first-probe blocker
by splitting it into residual runner_v3, bounded_wait_nofinal_v14 native, and
residual native completed-history service classes.  The bounded split is
necessary: the coarse native c9_16 class is not theorem-usable because its
slowest lower-service point does not dominate the aggregate class load.
Module72 closes the CFCMT portion of `sumo_eval_cpu|c_le2` with six script-level
completed-history service classes: feed conversion, environment validation,
SUMO generation, snapshot generation, policy rollout, and traffic-signal
phase2.  It intentionally leaves offline-sumo, H2Oplus, RESCO/config,
ZSW metrics-parser, and Nature-emissions SUMO rows unclaimed.
Module73 removes the non-schedulable external auto-adopted stdin/wait-for
processes from the theorem-facing production population.  This is an
operational-semantics correction, not a service measurement: these records have
no scheduler id, no scheduler log, and no reproducible progress unit, so they
cannot be part of the controlled arrival process.
Module74 closes the previous `freqduet_cpu_ablation|c_17_32` first-probe
residual by splitting direct runner_v3, paper-longtrain shell-wrapper, and
native-promotion residual records into separate completed-history service
classes.
Module75 closes the previous `bamor_cpu_training|c_9_16` first-probe blocker by
splitting BAMOR CPU-training records into train-compare, Mujoco, and
diagnostic-shard script-level service classes.  This is a service-semantics
refinement as well as a coverage gain: the diagnostic-shard rate is orders of
magnitude higher than train-compare, so one coarse BAMOR c9_16 class would be
both conservative and misleading.
Module76 closes most of the previous `freqduet_cpu_ablation|c_le2` residual by
splitting FreqDuet ablation, baseline-rule, preflight/env-check, Transit/FreqHRL
analysis-matrix, and Transit/FreqHRL merge jobs into five completed-history
service classes.  Two `freqduet_autoadopt_spin.py` helpers remain unmeasured.

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
84 c_le2 direct `runner_v3.py` records.  Module63 maps 91 completed-active
Transit native-promotion c17_32 seed-range validation records using parsed
seed-episode units.  Module64 introduced the broad BAMOR c3_8 CPU-training
completed-history certificate; Module67 refines it into train-compare, Mujoco,
and diagnostic-shard script-level classes.  Module65 maps 50 completed-active
ZSW TSP/SUMO c_le2 runner records using parsed simulated-second units.  Module66
maps 86 c3_8 direct `runner_v3.py` FreqDuet records.  Module68 maps 55
c33_64 native-promotion batch records with parsed seed-episode units.  Module69
maps 49 c65p native-promotion records, 7 c65p `run_freqduet_ablation.py`
records, and 6 c65p promoted ep100 shell-batch records.  Module70 maps 19 c_le2
native-promotion records, 7 native-control records, 18 trading-policy records,
14 trading-sweep records, 3 surrogate-validation records, and 1 import-smoke
singleton.  Module71 maps 40 residual c9_16 native-promotion records, 7
bounded-wait c9_16 native-promotion records, and 12 residual c9_16 runner_v3
records.  Module72 maps 10 CFCMT policy-rollout records, 5 feed-conversion
records, 5 environment-validation records, 5 snapshot-generation records, 4
SUMO-generation records, and 1 traffic-signal phase2 singleton.  Module74 maps
16 c17_32 runner_v3 records, 16 paper-longtrain shards, and 14 residual
native-promotion records.  Module75 maps 34 c9_16 BAMOR diagnostic-shard
records, 6 c9_16 BAMOR train-compare records, and 3 c9_16 BAMOR Mujoco records
with script-specific training-step lower-service points.  Module76 maps 11
Transit/FreqHRL analysis-matrix records, 6 Transit/FreqHRL merge records, 5
FreqDuet preflight records, 5 FreqDuet baseline-rule records, and 3 FreqDuet
c_le2 ablation records.  The next
closure targets are now the remaining CPU/SUMO/transit residual buckets, led by
`bamor_cpu_training|c_le2`, `sumo_eval_cpu|c_le2`, and
`freqduet_cpu_ablation|c_3_8` in the regenerated Module53 manifest.
