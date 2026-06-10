# Module53 CPU/SUMO/Transit Probe Manifest

Date: 2026-06-11

Module51 identified `cpu_sumo_transit_eval_or_control` as the dominant
production-coverage blocker.  Module53 breaks that bucket into concrete
sub-buckets from real Scheduleurm task records, so the next experiments are not
ambiguous.

## Artifacts

```text
algorithm/experiments/production_bucket_probe_manifest.py
md/experiment_artifacts/module53_cpu_sumo_transit_probe_manifest.json
md/experiment_artifacts/module53_cpu_sumo_transit_probe_manifest.md
```

## Status After Module86

The original top sub-bucket was:

```text
freqduet_cpu_ablation|c_17_32
```

Module56 measured the exact `run_freqduet_ablation.py` slice inside that
sub-bucket on `jtl110cpu2`, loaded feasible profiles `1,2,4` into the service
cache, and loaded profile `8` as a capacity boundary.  That exact command slice
is now mapped strictly as:

```text
workload_key = freqduet_cpu_ablation_c17_32
completed_active_production mapped count = 147
raw_history_all mapped count = 173
```

Module57 additionally measured one exact direct-runner config inside c9_16:

```text
workload_key = freqduet_runner_v3_allfreq_alllayers_c9_16
completed_active_production mapped count = 1
raw_history_all mapped count = 1
feasible profiles = 1,2,4,8
```

Module58 then measured the dominant `run_freqduet_ablation.py` command shape
inside c9_16:

```text
workload_key = freqduet_cpu_ablation_c9_16
completed_active_production mapped count = 110
raw_history_all mapped count = 129
feasible profiles = 1,2,4,8
strict unit rule = parsed jobs times episodes
```

Module59 measured a conservative completed-history profile-1 lower-service
certificate for clean SimpleSAC c_le2 eval tasks:

```text
workload_key = sumo_eval_simple_sac_c_le2
completed_active_production mapped count = 54
raw_history_all mapped count = 71
feasible profiles = 1
unit = eval_json
```

Module60 then measured a conservative completed-history profile-1 lower-service
certificate for parseable c3_8 FreqDuet ablation tasks:

```text
workload_key = freqduet_cpu_ablation_c3_8_completed_history
completed_active_production mapped count = 40
raw_history_all mapped count = 42
feasible profiles = 1
unit = episode
```

Module61 measured the analogous completed-history profile-1 lower-service
certificate for parseable c33_64 FreqDuet ablation tasks:

```text
workload_key = freqduet_cpu_ablation_c33_64_completed_history
completed_active_production mapped count = 63
raw_history_all mapped count = 63
feasible profiles = 1
unit = episode
```

Module62 measured a completed-history profile-1 lower-service portfolio for
direct c_le2 FreqDuet runner tasks:

```text
workload_key = freqduet_runner_v3_c_le2_completed_history
completed_active_production mapped count = 84
raw_history_all mapped count = 84
feasible profiles = 1
unit = episode
```

Module63 measured a completed-history profile-1 lower-service point for Transit
native-promotion validation records inside the c17_32 residual bucket when the
command carries explicit seed-index ranges:

```text
workload_key = transit_native_promotion_c17_32_seedrange_completed_history
completed-history service records used = 71
current completed_active_production mapped count = 91
current raw_history_all mapped count = 153
feasible profiles = 1
unit = seed_episode
```

Module64 measured a completed-history profile-1 lower-service point for BAMOR
c3_8 CPU training records with parseable training-step units.  Module67 then
refined this broad class into three script-level certificates:

```text
bamor_train_compare_c3_8_completed_history: completed_active = 58, raw_history_all = 68
bamor_mujoco_c3_8_completed_history: completed_active = 123, raw_history_all = 137
bamor_diagnostic_shard_c3_8_completed_history: completed_active = 25, raw_history_all = 25
feasible profiles = 1
unit = training_step
```

Module65 measured a completed-history profile-1 lower-service point for ZSW
TSP/SUMO c_le2 runner records with explicit simulated duration:

```text
workload_key = zsw_tsp_sumo_eval_c_le2_completed_history
completed_active_production mapped count = 50
raw_history_all mapped count = 50
feasible profiles = 1
unit = sim_second
```

Module66 measured a completed-history profile-1 lower-service point for direct
c3_8 FreqDuet runner tasks:

```text
workload_key = freqduet_runner_v3_c3_8_completed_history
completed_active_production mapped count = 86
raw_history_all mapped count = 86
feasible profiles = 1
unit = episode
```

Module68 measured completed-history profile-1 lower-service points for c33_64
native-promotion records with parseable seed work units, split into batch and
single-seed classes:

```text
workload_key = transit_native_promotion_c33_64_batch_completed_history
completed-history service records used = 45
current completed_active_production mapped count = 55
current raw_history_all mapped count = 179
feasible profiles = 1
unit = seed_episode

workload_key = transit_native_promotion_c33_64_single_seed_completed_history
completed_history mapped count = 2
feasible profiles = 1
unit = seed_episode
```

Module69 measured completed-history profile-1 lower-service points for the
previous c65p top residual by splitting it into three command-shape classes:

```text
workload_key = freqduet_cpu_ablation_c65p_completed_history
completed_active_production mapped count = 7
raw_history_all mapped count = 11
feasible profiles = 1
unit = episode

workload_key = freqduet_promoted_ep100_c65p_completed_history
completed_active_production mapped count = 6
raw_history_all mapped count = 6
feasible profiles = 1
unit = episode

workload_key = transit_native_promotion_c65p_completed_history
completed-history service records used = 48
current completed_active_production mapped count = 49
current raw_history_all mapped count = 115
feasible profiles = 1
unit = seed_episode
```

Module70 measured completed-history profile-1 lower-service points for low-CPU
Transit/FreqHRL validation and control records, split by script semantics:

```text
transit_trading_sweep_c_le2_completed_history: completed_active = 14, unit = market_step
transit_trading_policy_c_le2_completed_history: completed_active = 18, unit = policy_market_step
transit_surrogate_validation_c_le2_completed_history: completed_active = 3, unit = surrogate_corridor_step
transit_native_promotion_c_le2_completed_history: completed_active = 19, unit = native_variant_episode
transit_native_control_c_le2_completed_history: completed_active = 7, unit = native_control_episode
transit_freqhrl_import_smoke_c_le2_completed_history: completed_active = 1, unit = import_check
```

This removes the former `transit_freqhrl_cpu_validation|c_le2` top residual
without merging trading, surrogate, native simulator, and import-smoke work.

Module71 measured completed-history profile-1 lower-service points for the
previous c9_16 residual command shapes.  The first coarse attempt merged all
native-promotion c9_16 records into one service class and was rejected because
the mapped production-load LP became infeasible.  The accepted certificate
therefore splits the slice by service semantics:

```text
transit_native_promotion_c9_16_bounded_wait_completed_history: completed_active = 7, raw_history_all = 9, unit = seed_episode, min = 0.003792221 seed-episode/s
transit_native_promotion_c9_16_residual_completed_history: completed_active = 40, raw_history_all = 45, unit = seed_episode, min = 0.035084877 seed-episode/s
freqduet_runner_v3_c9_16_residual_completed_history: completed_active = 12, raw_history_all = 13, unit = episode, min = 0.030459670 episode/s
```

This removes the previous `freqduet_cpu_ablation|c_9_16` first-probe blocker
without charging the slow bounded-wait stress profile to faster residual native
or direct-runner work.

Module72 measured completed-history profile-1 lower-service points for the
CFCMT portion of the low-CPU SUMO/evaluation bucket.  It splits the 30
completed-active CFCMT records into six script-level classes:

```text
cfcmt_feed_conversion_c_le2_completed_history: completed_active = 5, unit = feed_conversion, min = 0.000676721 feed_conversion/s
cfcmt_env_validation_c_le2_completed_history: completed_active = 5, unit = validation_step, min = 11.792971439 validation-step/s
cfcmt_sumo_generation_c_le2_completed_history: completed_active = 4, unit = sim_second, min = 18.527090541 sim-second/s
cfcmt_snapshot_generation_c_le2_completed_history: completed_active = 5, unit = snapshot_window, min = 0.028279002 snapshot-window/s
cfcmt_policy_rollout_c_le2_completed_history: completed_active = 10, unit = policy_event_budget, min = 10.848108919 policy-event-budget/s
cfcmt_traffic_signal_phase2_c_le2_completed_history: completed_active = 1, unit = phase2_run, min = 0.001201620 phase2-run/s
```

The split is necessary because policy-rollout and snapshot commands reference
`sumo_generation` outputs in paths, but their progress units are not
SUMO-generation simulated seconds.

The current Module53 manifest has therefore been regenerated over the remaining
unmeasured `cpu_sumo_transit_eval_or_control` records.

Module73 then tightened the theorem-facing production population by excluding
unobservable external auto-adopted stdin/wait-for processes: no scheduler id,
no scheduler log, no reproducible command template, and no progress unit.  This
removes the former `transit_misc_cpu|c_le2` first-probe blocker from the
controlled-arrival theorem population instead of pretending it has a measured
Transit service curve.

Module74 measured completed-history profile-1 lower-service points for the
previous c17_32 residual command shapes:

```text
freqduet_runner_v3_c17_32_completed_history: completed_active = 16, unit = episode, min = 0.017633513 episode/s
freqduet_paper_longtrain_c17_32_completed_history: completed_active = 16, unit = longtrain_shard, min = 0.000064307 shard/s
transit_native_promotion_c17_32_residual_completed_history: completed_active = 14, unit = seed_episode, min = 0.059097464 seed-episode/s
```

The paper-longtrain class intentionally uses completed shard as its progress
unit because the production shell commands used `--skip-existing`; counting
job-count times episodes would overstate service if prior outputs were reused.

Module75 measured completed-history profile-1 lower-service points for the
previous BAMOR c9_16 CPU-training first-probe blocker, split by script
semantics:

```text
bamor_train_compare_c9_16_completed_history: completed_active = 6, raw_history_all = 6, unit = training_step, min = 1.507946928 training-step/s
bamor_mujoco_c9_16_completed_history: completed_active = 3, raw_history_all = 3, unit = training_step, min = 37.690625799 training-step/s
bamor_diagnostic_shard_c9_16_completed_history: completed_active = 34, raw_history_all = 34, unit = training_step, min = 1051.983312651 training-step/s
```

The split prevents the slow train-compare lower-service point from being used
as the service class for diagnostic-shard tasks, and it prevents the fast
diagnostic-shard rate from being charged to slower train-compare work.

Module76 measured completed-history profile-1 lower-service points for five
c_le2 residual command shapes that had previously been grouped under
`freqduet_cpu_ablation|c_le2`:

```text
freqduet_cpu_ablation_c_le2_completed_history: completed_active = 3, raw_history_all = 7, unit = episode, min = 0.013767135 episode/s
freqduet_baseline_rule_c_le2_completed_history: completed_active = 5, raw_history_all = 5, unit = episode, min = 0.279960678 episode/s
freqduet_preflight_c_le2_completed_history: completed_active = 5, raw_history_all = 14, unit = preflight_check, min = 0.005813124 preflight-check/s
transit_freqhrl_analysis_matrix_c_le2_completed_history: completed_active = 11, raw_history_all = 14, unit = analysis_job, min = 0.005627378 analysis-job/s
transit_freqhrl_merge_c_le2_completed_history: completed_active = 6, raw_history_all = 8, unit = merge_job, min = 0.006077690 merge-job/s
```

Two `freqduet_autoadopt_spin.py` helpers remain unmeasured because their stable
progress unit is not part of the scheduler-controlled workload model.

Module77 measured completed-history profile-1 lower-service points for the
previous BAMOR c_le2 CPU-training first-probe blocker, split by script
semantics.  Some records specify `--device cuda` while the scheduler record has
zero estimated VRAM, so these are production-history service certificates rather
than generic CPU-only microbenchmark curves:

```text
bamor_train_compare_c_le2_completed_history: completed_active = 7, raw_history_all = 7, unit = training_step, min = 20.305242828 training-step/s
bamor_mujoco_c_le2_completed_history: completed_active = 16, raw_history_all = 16, unit = training_step, min = 23.686028832 training-step/s
bamor_diagnostic_shard_c_le2_completed_history: completed_active = 3, raw_history_all = 3, unit = training_step, min = 101.339501734 training-step/s
```

Module78 measured completed-history profile-1 lower-service points for the
previous `sumo_eval_cpu|c_le2` first-probe blocker.  Each class uses one
completed production command as its progress unit, rather than expanding
resume/skip-existing/shell-loop commands into seed or episode counts:

```text
offline_sumo_eval_c_le2_completed_history: completed_active = 9, raw_history_all = 11, unit = eval_job, min = 0.000012679 eval-job/s
h2oplus_shell_eval_c_le2_completed_history: completed_active = 4, raw_history_all = 5, unit = shell_eval_job, min = 0.000079624 shell-eval-job/s
zsw_metrics_parser_c_le2_completed_history: completed_active = 1, raw_history_all = 1, unit = metrics_parse_job, min = 0.058559362 metrics-parse-job/s
resco_config_eval_c_le2_completed_history: completed_active = 5, raw_history_all = 5, unit = resco_config_run, min = 0.000073785 resco-config-run/s
nature_emissions_extract_c_le2_completed_history: completed_active = 1, raw_history_all = 1, unit = nature_extract_job, min = 0.022218150 nature-extract-job/s
nature_emissions_sumo_c_le2_completed_history: completed_active = 1, raw_history_all = 1, unit = nature_sumo_run, min = 0.003339202 nature-sumo-run/s
```

Module79 measured completed-history profile-1 lower-service points for the
previous `freqduet_cpu_ablation|c_3_8` first-probe blocker.  The residual records
were Transit/FreqHRL native commands rather than FreqDuet ablations, so they are
split by native service semantics:

```text
transit_native_promotion_c3_8_persistent_stress_completed_history: completed_active = 7, raw_history_all = 7, unit = seed_episode, min = 0.008254466 seed-episode/s
transit_native_real_demand_batch_c3_8_completed_history: completed_active = 7, raw_history_all = 28, unit = native_control_episode, min = 0.013410918 native-control-episode/s
transit_native_real_demand_alighting_c3_8_completed_history: completed_active = 7, raw_history_all = 14, unit = native_control_episode, min = 0.019432989 native-control-episode/s
```

Module80 measured completed-history profile-1 lower-service points for the
previous `bamor_cpu_training|c_17_32` first-probe blocker.  The observed
production command shapes are c17_32 Mujoco and diagnostic-shard jobs, so the
certificate keeps them separate:

```text
bamor_mujoco_c17_32_completed_history: completed_active = 3, raw_history_all = 3, unit = training_step, min = 25.959797537 training-step/s
bamor_diagnostic_shard_c17_32_completed_history: completed_active = 14, raw_history_all = 14, unit = training_step, min = 896.623644960 training-step/s
```

Module81 measured a completed-history profile-1 lower-service point for direct
FreqDuet `runner_v3.py` records inside c33_64:

```text
freqduet_runner_v3_c33_64_completed_history: completed_active = 15, raw_history_all = 32, unit = episode, min = 0.028461965 episode/s
```

Module82 measured completed-history profile-1 lower-service points for the
previous `sumo_eval_cpu|c_3_8` blocker:

```text
cfcmt_snapshot_generation_c3_8_completed_history: completed_active = 1, raw_history_all = 1, unit = snapshot_window, min = 0.297007940 snapshot-window/s
cfcmt_pytest_sumo_c3_8_completed_history: completed_active = 4, raw_history_all = 4, unit = pytest_job, min = 0.013694244 pytest-job/s
cfcmt_traffic_signal_phase1_c3_8_completed_history: completed_active = 1, raw_history_all = 1, unit = phase1_run, min = 0.012971998 phase1-run/s
zsw_m21_sumo_eval_c3_8_completed_history: completed_active = 6, raw_history_all = 6, unit = sim_second, min = 7.269619812 sim-second/s
```

Module83 measured completed-history profile-1 lower-service points for the
Transit/FreqHRL c3_8 residual:

```text
transit_trading_public_csv_c3_8_completed_history: completed_active = 1, raw_history_all = 4, unit = csv_step, min = 59.878105731 csv-step/s
transit_trading_pressure_merge_c3_8_completed_history: completed_active = 1, raw_history_all = 3, unit = merge_job, min = 0.062236279 merge-job/s
transit_trading_policy_c3_8_completed_history: completed_active = 1, raw_history_all = 1, unit = policy_seed_step_asset, min = 398.625038290 policy-seed-step-asset/s
transit_surrogate_c3_8_completed_history: completed_active = 1, raw_history_all = 1, unit = surrogate_seed_step_corridor, min = 299.115070129 surrogate-seed-step-corridor/s
transit_freqhrl_tests_c3_8_completed_history: completed_active = 8, raw_history_all = 8, unit = test_job, min = 0.010963127 test-job/s
transit_native_merge_c3_8_completed_history: completed_active = 1, raw_history_all = 1, unit = native_merge_job, min = 0.012513107 native-merge-job/s
```

Module84 then closes the Transit/FreqHRL c17_32 residual with four
completed-history lower-service rows:

```text
transit_trading_pressure_matrix_c17_32_completed_history: completed_active = 1, raw_history_all = 5, unit = seed_step_asset_scenario_baseline, min = 6526.878819304 seed-step-asset-scenario-baseline/s
transit_trading_promotion_recovery_c17_32_completed_history: completed_active = 1, raw_history_all = 1, unit = recovery_command, min = 0.163527120 recovery-command/s
transit_demand_estimator_c17_32_completed_history: completed_active = 2, raw_history_all = 10, unit = seed_step, min = 225.220667384 seed-step/s
transit_gap_closure_c17_32_completed_history: completed_active = 2, raw_history_all = 8, unit = surrogate_seed_step_corridor, min = 257.874955538 surrogate-seed-step-corridor/s
```

Module85 then closes the c9_16 wait-credit native-promotion shell residual:

```text
transit_native_promotion_c9_16_wait_credit_shell_completed_history: completed_active = 6, raw_history_all = 6, unit = seed_episode, min = 0.115357646 seed-episode/s
```

Module86 then closes the offline-sumo c33_64 eval-command residual:

```text
offline_sumo_eval_c33_64_completed_history: completed_active = 5, raw_history_all = 15, unit = eval_command, min = 0.000151576 eval-command/s
```

## Current Remaining Bucket

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 10
cpu_cores median = 4.500000
cpu_cores p90 = 48
cpu_cores max = 61
ram_mb median = 528.000
ram_mb p90 = 64000
ram_mb max = 64000
theorem_status = measurement_required
```

## Top Remaining Sub-Buckets

| Sub-Bucket | Count | Fraction |
|---|---:|---:|
| `freqduet_cpu_ablation|c_le2` | 3 | 0.300000 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.200000 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.200000 |
| `transit_freqhrl_cpu_validation|c_le2` | 2 | 0.200000 |
| `bamor_cpu_training|c_3_8` | 1 | 0.100000 |

## Next Probe Order

The regenerated manifest recommends this first pass over the remaining bucket:

```text
freqduet_cpu_ablation|c_le2
transit_freqhrl_cpu_validation|c_33_64
transit_freqhrl_cpu_validation|c_9_16
transit_freqhrl_cpu_validation|c_le2
bamor_cpu_training|c_3_8
```

Each remaining sub-bucket still needs progress-bearing service curves over:

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = local_cpu, direct_hpc_cpu_node
```

## Interpretation

Module53 no longer says the first production CPU/SUMO measurement is merely a
plan.  The exact `run_freqduet_ablation.py` c17_32 slice is measured and
strictly mapped, and one exact c9_16 direct-runner config is measured and
strictly mapped.  Module58 maps 110 c9_16 `run_freqduet_ablation.py` records
with parsed units.  Module59 maps 54 clean SimpleSAC c_le2 eval records with a
profile-1 completed-history lower-service point.  Module60 maps 40 c3_8
`run_freqduet_ablation.py` records with a profile-1 completed-history
lower-service point.  Module61 maps 63 c33_64 `run_freqduet_ablation.py`
records with a profile-1 completed-history lower-service point.  Module62 maps
84 c_le2 direct `runner_v3.py` records with a profile-1 completed-history
lower-service point.  Module63 uses 71 completed-history Transit
native-promotion c17_32 seed-range service records and now maps 91
completed-active records with that profile-1 lower-service point.  Module64
introduced the broad BAMOR c3_8 CPU-training completed-history point; Module67
refines it into 58 train-compare, 123 Mujoco, and 25 diagnostic-shard
completed-active records.  Module65 maps 50
completed-active ZSW TSP/SUMO c_le2 runner records with a profile-1
completed-history lower-service point.  Module66 maps 86 c3_8 direct
`runner_v3.py` records with a profile-1 completed-history lower-service point.
Module68 uses 45 c33_64 native-promotion completed-history batch service
records and now maps 55 completed-active records while keeping single-seed
smoke/fix records separate.  Module69 uses 48 c65p native-promotion completed
service records and now maps 49 completed-active c65p native-promotion records,
7 c65p `run_freqduet_ablation.py` records, and 6 c65p promoted ep100
shell-batch records with separated profile-1 completed-history lower-service
points.
Module70 maps the low-CPU Transit/FreqHRL validation/control slice into six
separate profile-1 completed-history service classes.  Module71 maps 40
residual c9_16 native-promotion records, 7 bounded-wait c9_16 native-promotion
records, and 12 residual c9_16 `runner_v3.py` records after rejecting the
infeasible coarse native c9_16 class.  Module72 maps 30 CFCMT low-CPU SUMO/eval
records into six script-level completed-history service classes.  Module73
removes unobservable external auto-adopted stdin/wait-for processes from the
controlled-arrival theorem population.  Module74 closes the previous c17_32
residual by splitting runner_v3, paper-longtrain, and native residual work.
Module75 closes the previous BAMOR c9_16 first-probe blocker by splitting
train-compare, Mujoco, and diagnostic-shard work into separate profile-1
completed-history lower-service classes.
Module76 closes most of the previous c_le2 residual by splitting FreqDuet
ablation, baseline-rule, preflight/env-check, Transit/FreqHRL analysis-matrix,
and Transit/FreqHRL merge work into five strict completed-history classes.
Module77 closes the previous BAMOR c_le2 first-probe blocker by splitting
train-compare, Mujoco, and diagnostic-shard work into separate profile-1
completed-history training-step classes.  Module78 closes the previous
`sumo_eval_cpu|c_le2` first-probe blocker with six completed-command service
classes.  Module79 closes the previous `freqduet_cpu_ablation|c_3_8` first-probe
blocker by mapping Transit/FreqHRL native commands to native service classes.
Module80 closes the previous `bamor_cpu_training|c_17_32` first-probe blocker
by mapping c17_32 BAMOR Mujoco and diagnostic-shard commands to separate
training-step completed-history service classes.
Module81 closes the direct-runner portion of the previous
`freqduet_cpu_ablation|c_33_64` blocker by mapping c33_64 FreqDuet
`runner_v3.py` commands to a separate episode completed-history service class.
Module82 closes the previous `sumo_eval_cpu|c_3_8` first-probe blocker by
splitting it into CFCMT snapshot generation, CFCMT SUMO/traffic pytest, CFCMT
traffic-signal phase1, and ZSW M21 completed-history service classes.  It does
not claim CFCMT local snapshot validation or non-M21 ZSW c3_8 command shapes.
Module83 closes the Transit/FreqHRL c3_8 residual by splitting public CSV eval,
pressure-matrix merge, policy-entry train/eval, PPO surrogate train/eval,
pytest/unittest jobs, and native-promotion shard merge into six service
classes.  Failed/cancelled same-shape attempts are accounted for in raw load
classification but are not used as lower-service samples.
Module84 closes the Transit/FreqHRL c17_32 residual by splitting pressure-test
matrix, promotion-recovery validation, demand-estimator validation, and
gap-closure validation into four service classes.  Failed/cancelled same-shape
attempts are accounted for in raw load classification but are not used as
lower-service samples.
Module85 closes the c9_16 wait-credit native-promotion shell residual with a
static shell arithmetic seed-range parser and a profile-1 lower-service row.
Module86 closes the offline-sumo c33_64 eval-command residual with one completed
eval command as the conservative unit; the `sumo_eval_cpu|c_33_64` sub-bucket is
therefore removed from the current first-probe order.
The global theorem remains open because 10 completed/active production records
in the CPU/SUMO/transit family still require measured curves or equivalence
certificates, led by `freqduet_cpu_ablation|c_le2`, two c33/c9/low-CPU
Transit/FreqHRL residual buckets, and one BAMOR c3_8 residual.
