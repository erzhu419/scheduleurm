# Module51 Production Coverage Drilldown

Date: 2026-06-11

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

## Current Result After Module88

For the 30-day completed/active production window:

```text
completed_active_production records = 3327
representative mapped = 2931
strict mapped = 1898
unmapped / measurement-required = 396
representative mapped_fraction = 0.880974
capacity slack on mapped raw-window strict load from Module49 delta = 0.000011136
global theorem closed = false
```

Module73 is not a service-curve shortcut.  It tightens the theorem-facing
population: external `auto-adopted` stdin or `scheduler.py wait-for` processes
with no scheduler id and no log are not scheduler-controlled arrivals.  They
remain operational telemetry/background-load evidence, but they are outside the
arrival stream that Scheduleurm can assign to service actions.

The generated artifact
`md/experiment_artifacts/module51_production_coverage_drilldown.md` contains the
full current strict mapped-key table.  The Module78 additions are:

```text
offline_sumo_eval_c_le2_completed_history = 9 / 2910 completed-active production records
h2oplus_shell_eval_c_le2_completed_history = 4 / 2910 completed-active production records
resco_config_eval_c_le2_completed_history = 5 / 2910 completed-active production records
zsw_metrics_parser_c_le2_completed_history = 1 / 2910 completed-active production records
nature_emissions_extract_c_le2_completed_history = 1 / 2910 completed-active production records
nature_emissions_sumo_c_le2_completed_history = 1 / 2910 completed-active production records
```

Module79 then adds:

```text
transit_native_promotion_c3_8_persistent_stress_completed_history = 7 / 3058 completed-active production records
transit_native_real_demand_batch_c3_8_completed_history = 7 / 3058 completed-active production records
transit_native_real_demand_alighting_c3_8_completed_history = 7 / 3058 completed-active production records
```

Module80 then adds:

```text
bamor_diagnostic_shard_c17_32_completed_history = 14 / 3101 completed-active production records
bamor_mujoco_c17_32_completed_history = 3 / 3101 completed-active production records
```

Module81 then adds:

```text
freqduet_runner_v3_c33_64_completed_history = 15 / 3095 completed-active production records
```

Module82 then adds:

```text
cfcmt_snapshot_generation_c3_8_completed_history = 1 / 3136 completed-active production records
cfcmt_pytest_sumo_c3_8_completed_history = 4 / 3136 completed-active production records
cfcmt_traffic_signal_phase1_c3_8_completed_history = 1 / 3136 completed-active production records
zsw_m21_sumo_eval_c3_8_completed_history = 6 / 3136 completed-active production records
```

Module83 then adds:

```text
transit_trading_public_csv_c3_8_completed_history = 1 / 3199 completed-active production records
transit_trading_pressure_merge_c3_8_completed_history = 1 / 3199 completed-active production records
transit_trading_policy_c3_8_completed_history = 1 / 3199 completed-active production records
transit_surrogate_c3_8_completed_history = 1 / 3199 completed-active production records
transit_freqhrl_tests_c3_8_completed_history = 8 / 3199 completed-active production records
transit_native_merge_c3_8_completed_history = 1 / 3199 completed-active production records
```

Module84 then adds:

```text
transit_trading_pressure_matrix_c17_32_completed_history = 1 / 3214 completed-active production records
transit_trading_promotion_recovery_c17_32_completed_history = 1 / 3214 completed-active production records
transit_demand_estimator_c17_32_completed_history = 2 / 3214 completed-active production records
transit_gap_closure_c17_32_completed_history = 2 / 3214 completed-active production records
```

Module85 then adds:

```text
transit_native_promotion_c9_16_wait_credit_shell_completed_history = 6 / 3211 completed-active production records
```

Module86 then adds:

```text
offline_sumo_eval_c33_64_completed_history = 5 / 3255 completed-active production records
```

Module87 then corrects the production-population boundary:

```text
external freqduet_autoadopt_spin.py helpers excluded = 2
reason = external auto-adopted, no scheduler id, no scheduler log, no progress unit
```

Module88 then adds:

```text
transit_trading_policy_c33_64_completed_history = 1 / 3327 completed-active production records
transit_trading_pressure_matrix_c33_64_completed_history = 1 / 3327 completed-active production records
```

Module51 now intentionally reports coverage only.  The mapped representative
raw-window load still has positive capacity slack in Module49, but that does
not close the global theorem because measured coverage is still incomplete.

## Largest Remaining Buckets

| Bucket | Status | Count | Fraction |
|---|---|---:|---:|
| `generic_cpu_python` | measurement required | 162 | 0.048693 |
| `cpu_eval_generic` | measurement required | 82 | 0.024647 |
| `artifact_io_control` | measurement required | 77 | 0.023144 |
| `scheduler_control_plane` | measurement required | 56 | 0.016832 |
| `gpu_rl_unmeasured_variant` | measurement required | 14 | 0.004208 |
| `cpu_sumo_transit_eval_or_control` | measurement required | 5 | 0.001503 |

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
after Module77:  cpu_sumo_transit_eval_or_control = 113 / 2766
after Module78:  cpu_sumo_transit_eval_or_control = 94 / 2910
after Module79:  cpu_sumo_transit_eval_or_control = 73 / 3058
after Module80:  cpu_sumo_transit_eval_or_control = 56 / 3101
after Module81:  cpu_sumo_transit_eval_or_control = 43 / 3095
after Module82:  cpu_sumo_transit_eval_or_control = 38 / 3136
after Module83:  cpu_sumo_transit_eval_or_control = 26 / 3199
after Module84:  cpu_sumo_transit_eval_or_control = 20 / 3214
after Module85:  cpu_sumo_transit_eval_or_control = 14 / 3211
after Module86:  cpu_sumo_transit_eval_or_control = 10 / 3255
after Module87:  cpu_sumo_transit_eval_or_control = 7 / 3280
after Module88:  cpu_sumo_transit_eval_or_control = 5 / 3327
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
Module77 closes the previous `bamor_cpu_training|c_le2` first-probe blocker by
splitting BAMOR c_le2 training records into train-compare, Mujoco, and
diagnostic-shard script-level service classes.  Several records carry
`--device cuda` while the scheduler estimated zero VRAM, so they are certified as
production-history service rows, not as generic CPU-only workload curves.
Module78 closes the previous `sumo_eval_cpu|c_le2` first-probe blocker by
splitting offline-sumo eval commands, H2Oplus/SimpleSAC complex shell eval jobs,
ZSW metrics parsing, RESCO config/main.py runs, and Nature-emissions
extraction/SUMO singletons.  The conservative unit is one completed production
command, not an expanded episode/seed/checkpoint count.
Module79 closes the previous `freqduet_cpu_ablation|c_3_8` first-probe blocker
by recognizing the remaining records as Transit/FreqHRL native work and mapping
persistent-stress native promotion, native real-demand batch validation, and
alighting-safe/rescue shard validation separately.
Module80 closes the previous `bamor_cpu_training|c_17_32` first-probe blocker
for observed production command shapes by splitting c17_32 BAMOR training into
Mujoco and diagnostic-shard completed-history training-step service classes.
It does not claim a c17_32 train-compare class because no such command shape is
present in the current production bucket.
Module81 closes the direct-runner portion of the previous
`freqduet_cpu_ablation|c_33_64` blocker by mapping c33_64 FreqDuet
`runner_v3.py` commands with explicit episode units to their own
completed-history service class.
Module82 closes the previous `sumo_eval_cpu|c_3_8` blocker by mapping CFCMT
snapshot generation, CFCMT SUMO/traffic pytest, CFCMT traffic-signal phase1,
and ZSW M21 runner records into four separate completed-history service
classes.  This split is required for service semantics: pytest jobs are not
snapshot-window work, and c3_8 ZSW baseline/oracle runners are not claimed by
the M21 certificate.
Module83 closes the next Transit/FreqHRL c3_8 residual by mapping public CSV
market evaluation, pressure-matrix merge, policy-entry train/eval, PPO
surrogate train/eval, pytest/unittest jobs, and native-promotion shard merge
into six separate completed-history service classes.
Module84 closes the next Transit/FreqHRL c17_32 residual by mapping pressure
matrix, promotion-recovery validation, demand-estimator validation, and
gap-closure validation into four separate completed-history service classes.
This does not claim all c17_32 Transit work; it claims only these four observed
command shapes with explicit parser semantics.
Module85 closes the next c9_16 native-promotion shell-shard residual by mapping
wait-credit v39 shell arithmetic shards into a separate completed-history
service class.  The parser statically evaluates literal integer assignments and
does not execute the shell command.
Module86 closes the offline-sumo c33_64 eval-command residual by mapping the
five completed-active high-CPU offline-sumo rerun shards into a separate
completed-history service class.  The unit is one completed production eval
command; `--skip_existing` item ranges are not expanded.
Module87 removes the two external `freqduet_autoadopt_spin.py` helpers from the
controlled-arrival theorem population.  This is not a service certificate; it is
the same no-scheduler-id/no-log/no-progress-unit population-boundary correction
used for unobservable external stdin/wait-for helpers.
Module88 closes the c33_64 Transit/FreqHRL trading residual by mapping the
policy-entry and pressure-test matrix command shapes into separate
completed-history service classes.

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
c_le2 ablation records.  Module77 maps 16 c_le2 BAMOR Mujoco records, 7 c_le2
BAMOR train-compare records, and 3 c_le2 BAMOR diagnostic-shard records.  Module78
maps 9 completed-active offline-sumo eval records, 4 H2Oplus/SimpleSAC shell eval
records, 5 RESCO config/main.py records, and three singleton parser/extraction/SUMO
records.  Module80 maps 14 c17_32 BAMOR diagnostic-shard records and 3 c17_32
BAMOR Mujoco records.  Module81 maps 15 completed-active c33_64 FreqDuet
direct runner records.  Module82 maps 12 completed-active SUMO c3_8 records
across CFCMT snapshot/pytest/phase1 and ZSW M21 service classes.  Module83 maps
13 completed-active Transit/FreqHRL c3_8 residual records across six service
classes.  Module84 maps 6 completed-active Transit/FreqHRL c17_32 residual
records across four service classes.  Module85 maps 6 completed-active c9_16
wait-credit shell shards.  Module86 maps 5 completed-active offline-sumo c33_64
eval-command shards.  Module87 excludes 2 external auto-adopt spin helpers from
the controlled-arrival population.  Module88 maps 2 completed-active c33_64
Transit/FreqHRL trading records.  The next closure targets are now the remaining
CPU/SUMO/transit residual buckets, led by
`transit_freqhrl_cpu_validation|c_9_16`,
`transit_freqhrl_cpu_validation|c_le2`, and `bamor_cpu_training|c_3_8` in the
regenerated Module53 manifest.
