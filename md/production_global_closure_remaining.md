# Production Global Closure Remaining

Date: 2026-06-10

This note records the current gap after Module79.  It should be read together
with:

```text
md/experiment_module53_cpu_sumo_transit_probe_manifest.md
md/experiment_module56_freqduet_cpu_production_curve.md
md/experiment_module57_freqduet_runner_v3_c9_16_curve.md
md/experiment_module58_freqduet_ablation_c9_16_curve.md
md/experiment_module59_simple_sac_sumo_eval_cle2_completed_history.md
md/experiment_module60_freqduet_ablation_c3_8_completed_history.md
md/experiment_module61_freqduet_ablation_c33_64_completed_history.md
md/experiment_module62_freqduet_runner_v3_c_le2_completed_history.md
md/experiment_module63_transit_native_promotion_c17_32_seedrange_completed_history.md
md/experiment_module64_bamor_cpu_training_c3_8_completed_history.md
md/experiment_module65_zsw_tsp_sumo_eval_c_le2_completed_history.md
md/experiment_module66_freqduet_runner_v3_c3_8_completed_history.md
md/experiment_module67_bamor_train_compare_c3_8_completed_history.md
md/experiment_module67_bamor_mujoco_c3_8_completed_history.md
md/experiment_module67_bamor_diagnostic_shard_c3_8_completed_history.md
md/experiment_module68_transit_native_promotion_c33_64_batch_completed_history.md
md/experiment_module68_transit_native_promotion_c33_64_single_seed_completed_history.md
md/experiment_module69_freqduet_ablation_c65p_completed_history.md
md/experiment_module69_freqduet_promoted_ep100_c65p_completed_history.md
md/experiment_module69_transit_native_promotion_c65p_completed_history.md
md/experiment_module70_transit_trading_sweep_c_le2_completed_history.md
md/experiment_module70_transit_trading_policy_c_le2_completed_history.md
md/experiment_module70_transit_surrogate_validation_c_le2_completed_history.md
md/experiment_module70_transit_native_promotion_c_le2_completed_history.md
md/experiment_module70_transit_native_control_c_le2_completed_history.md
md/experiment_module70_transit_freqhrl_import_smoke_c_le2_completed_history.md
md/experiment_module71_transit_native_promotion_c9_16_bounded_wait_completed_history.md
md/experiment_module71_transit_native_promotion_c9_16_residual_completed_history.md
md/experiment_module71_freqduet_runner_v3_c9_16_residual_completed_history.md
md/experiment_module72_cfcmt_feed_conversion_c_le2_completed_history.md
md/experiment_module72_cfcmt_env_validation_c_le2_completed_history.md
md/experiment_module72_cfcmt_sumo_generation_c_le2_completed_history.md
md/experiment_module72_cfcmt_snapshot_generation_c_le2_completed_history.md
md/experiment_module72_cfcmt_policy_rollout_c_le2_completed_history.md
md/experiment_module72_cfcmt_traffic_signal_phase2_c_le2_completed_history.md
md/experiment_module74_freqduet_runner_v3_c17_32_completed_history.md
md/experiment_module74_freqduet_paper_longtrain_c17_32_completed_history.md
md/experiment_module74_transit_native_promotion_c17_32_residual_completed_history.md
md/experiment_module75_bamor_train_compare_c9_16_completed_history.md
md/experiment_module75_bamor_mujoco_c9_16_completed_history.md
md/experiment_module75_bamor_diagnostic_shard_c9_16_completed_history.md
md/experiment_module76_freqduet_ablation_c_le2_completed_history.md
md/experiment_module76_freqduet_baseline_rule_c_le2_completed_history.md
md/experiment_module76_freqduet_preflight_c_le2_completed_history.md
md/experiment_module76_transit_freqhrl_analysis_matrix_c_le2_completed_history.md
md/experiment_module76_transit_freqhrl_merge_c_le2_completed_history.md
md/experiment_module77_bamor_train_compare_c_le2_completed_history.md
md/experiment_module77_bamor_mujoco_c_le2_completed_history.md
md/experiment_module77_bamor_diagnostic_shard_c_le2_completed_history.md
md/experiment_module78_offline_sumo_eval_c_le2_completed_history.md
md/experiment_module78_h2oplus_shell_eval_c_le2_completed_history.md
md/experiment_module78_zsw_metrics_parser_c_le2_completed_history.md
md/experiment_module78_resco_config_eval_c_le2_completed_history.md
md/experiment_module78_nature_emissions_extract_c_le2_completed_history.md
md/experiment_module78_nature_emissions_sumo_c_le2_completed_history.md
md/experiment_module79_transit_native_promotion_c3_8_persistent_stress_completed_history.md
md/experiment_module79_transit_native_real_demand_batch_c3_8_completed_history.md
md/experiment_module79_transit_native_real_demand_alighting_c3_8_completed_history.md
md/experiment_module51_production_coverage_drilldown.md
md/or_submission_gap_closure.md
```

## Current State

The global production theorem is not fully closed.

Already closed:

```text
exact slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_17_32
workload_key = freqduet_cpu_ablation_c17_32
completed-active strict mapped count = 147
feasible profiles = 1,2,4
capacity boundary = 8

exact slice = runner_v3.py --config configs_freqduet/F_allfreq_alllayers_hiro.yaml within c_9_16
workload_key = freqduet_runner_v3_allfreq_alllayers_c9_16
completed-active strict mapped count = 1
feasible profiles = 1,2,4,8

command-shape slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_9_16
workload_key = freqduet_cpu_ablation_c9_16
completed-active strict mapped count = 110
feasible profiles = 1,2,4,8
unit rule = parsed jobs times episodes

completed-history slice = clean SimpleSAC run_multiseed_eval.sh within sumo_eval_cpu|c_le2
workload_key = sumo_eval_simple_sac_c_le2
completed-active strict mapped count = 54
feasible profiles = 1
unit = eval_json

completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_3_8
workload_key = freqduet_cpu_ablation_c3_8_completed_history
completed-active strict mapped count = 40
feasible profiles = 1
unit rule = parsed jobs times episodes

completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_33_64
workload_key = freqduet_cpu_ablation_c33_64_completed_history
completed-active strict mapped count = 63
feasible profiles = 1
unit rule = parsed jobs times episodes

completed-history slice = runner_v3.py within freqduet_cpu_ablation|c_le2
workload_key = freqduet_runner_v3_c_le2_completed_history
completed-active strict mapped count = 84
feasible profiles = 1
unit rule = parsed episodes

completed-history slice = Transit native_promotion_replan_validation within c_17_32
workload_key = transit_native_promotion_c17_32_seedrange_completed_history
completed-active strict mapped count = 91
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = BAMOR c_3_8 train_compare_baselines.py
workload_key = bamor_train_compare_c3_8_completed_history
completed-active strict mapped count = 58
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_3_8 train_bamor_mujoco.py
workload_key = bamor_mujoco_c3_8_completed_history
completed-active strict mapped count = 123
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_3_8 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c3_8_completed_history
completed-active strict mapped count = 25
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = ZSW TSP/SUMO c_le2 runners
workload_key = zsw_tsp_sumo_eval_c_le2_completed_history
completed-active strict mapped count = 50
feasible profiles = 1
unit rule = parsed simulated SUMO seconds from --duration

completed-history slice = runner_v3.py within freqduet_cpu_ablation|c_3_8
workload_key = freqduet_runner_v3_c3_8_completed_history
completed-active strict mapped count = 86
feasible profiles = 1
unit rule = parsed episodes

completed-history slice = Transit native_promotion_replan_validation batch within c_33_64
workload_key = transit_native_promotion_c33_64_batch_completed_history
completed-active strict mapped count = 55
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = Transit native_promotion_replan_validation single-seed smoke/fix within c_33_64
workload_key = transit_native_promotion_c33_64_single_seed_completed_history
certificate record count = 2
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_65p
workload_key = freqduet_cpu_ablation_c65p_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed jobs times episodes

completed-history slice = run_freqduet_promoted_ep100_hpc_batch.sh within c_65p
workload_key = freqduet_promoted_ep100_c65p_completed_history
completed-active strict mapped count = 6
feasible profiles = 1
unit rule = parsed job-count times 100 episodes

completed-history slice = Transit native_promotion_replan_validation within c_65p
workload_key = transit_native_promotion_c65p_completed_history
completed-history service records used = 48
current completed-active strict mapped count = 49
feasible profiles = 1
unit rule = statically parsed seed-count times episodes

completed-history slice = Transit/FreqHRL trading sweep within c_le2
workload_key = transit_trading_sweep_c_le2_completed_history
completed-active strict mapped count = 14
feasible profiles = 1
unit rule = parsed market-step grid units

completed-history slice = Transit/FreqHRL trading policy within c_le2
workload_key = transit_trading_policy_c_le2_completed_history
completed-active strict mapped count = 18
feasible profiles = 1
unit rule = parsed train/eval policy-market-step units

completed-history slice = Transit surrogate validation within c_le2
workload_key = transit_surrogate_validation_c_le2_completed_history
completed-active strict mapped count = 3
feasible profiles = 1
unit rule = parsed surrogate corridor-step units

completed-history slice = Transit native_promotion_replan_validation within c_le2
workload_key = transit_native_promotion_c_le2_completed_history
completed-active strict mapped count = 19
feasible profiles = 1
unit rule = parsed native variant-episode units

completed-history slice = Transit native wait-credit / real-demand control within c_le2
workload_key = transit_native_control_c_le2_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed native control episode units

completed-history singleton = Transit/FreqHRL Windows IMPORT_OK check
workload_key = transit_freqhrl_import_smoke_c_le2_completed_history
completed-active strict mapped count = 1
feasible profiles = 1
unit rule = import_check

completed-history slice = bounded-wait Transit native_promotion_replan_validation within c_9_16
workload_key = transit_native_promotion_c9_16_bounded_wait_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = statically parsed seed-count times episodes

completed-history slice = residual Transit native_promotion_replan_validation within c_9_16
workload_key = transit_native_promotion_c9_16_residual_completed_history
completed-active strict mapped count = 40
feasible profiles = 1
unit rule = statically parsed seed-count times episodes

completed-history slice = residual runner_v3.py within freqduet_cpu_ablation|c_9_16
workload_key = freqduet_runner_v3_c9_16_residual_completed_history
completed-active strict mapped count = 12
feasible profiles = 1
unit rule = parsed episodes

completed-history slice = CFCMT GTFS/LTA feed conversion within c_le2
workload_key = cfcmt_feed_conversion_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = feed-conversion job

completed-history slice = CFCMT H2O environment validation within c_le2
workload_key = cfcmt_env_validation_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = parsed validation steps

completed-history slice = CFCMT SUMO/APC/AVL generation within c_le2
workload_key = cfcmt_sumo_generation_c_le2_completed_history
completed-active strict mapped count = 4
feasible profiles = 1
unit rule = parsed simulated seconds

completed-history slice = CFCMT SUMO/APC/AVL snapshot generation within c_le2
workload_key = cfcmt_snapshot_generation_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = inferred snapshot windows

completed-history slice = CFCMT policy rollout validation within c_le2
workload_key = cfcmt_policy_rollout_c_le2_completed_history
completed-active strict mapped count = 10
feasible profiles = 1
unit rule = policy-event budget

completed-history singleton = CFCMT traffic_signal_sumo_phase2.py
workload_key = cfcmt_traffic_signal_phase2_c_le2_completed_history
completed-active strict mapped count = 1
feasible profiles = 1
unit rule = phase2_run

completed-history slice = BAMOR c_9_16 train_compare_baselines.py
workload_key = bamor_train_compare_c9_16_completed_history
completed-active strict mapped count = 6
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_9_16 train_bamor_mujoco.py
workload_key = bamor_mujoco_c9_16_completed_history
completed-active strict mapped count = 3
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_9_16 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c9_16_completed_history
completed-active strict mapped count = 34
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = FreqDuet c_le2 run_freqduet_ablation.py
workload_key = freqduet_cpu_ablation_c_le2_completed_history
completed-active strict mapped count = 3
feasible profiles = 1
unit rule = parsed jobs times episodes

completed-history slice = FreqDuet c_le2 run_baseline_rule.py
workload_key = freqduet_baseline_rule_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = parsed episodes

completed-history slice = FreqDuet c_le2 preflight/env checks
workload_key = freqduet_preflight_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = preflight_check

completed-history slice = Transit/FreqHRL c_le2 analysis matrix/report jobs
workload_key = transit_freqhrl_analysis_matrix_c_le2_completed_history
completed-active strict mapped count = 11
feasible profiles = 1
unit rule = analysis_job

completed-history slice = Transit/FreqHRL c_le2 shard merge jobs
workload_key = transit_freqhrl_merge_c_le2_completed_history
completed-active strict mapped count = 6
feasible profiles = 1
unit rule = merge_job

completed-history slice = BAMOR c_le2 train_compare_baselines.py
workload_key = bamor_train_compare_c_le2_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_le2 train_bamor_mujoco.py
workload_key = bamor_mujoco_c_le2_completed_history
completed-active strict mapped count = 16
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_le2 run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c_le2_completed_history
completed-active strict mapped count = 3
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = offline-sumo c_le2 eval_*.py production commands
workload_key = offline_sumo_eval_c_le2_completed_history
completed-active strict mapped count = 9
feasible profiles = 1
unit rule = one completed eval command

completed-history slice = H2Oplus/SimpleSAC c_le2 complex shell eval jobs
workload_key = h2oplus_shell_eval_c_le2_completed_history
completed-active strict mapped count = 4
feasible profiles = 1
unit rule = one completed shell eval job

completed-history singleton = ZSW c_le2 m1_metrics_parser.py
workload_key = zsw_metrics_parser_c_le2_completed_history
completed-active strict mapped count = 1
feasible profiles = 1
unit rule = one completed metrics parser job

completed-history slice = RESCO config/main.py c_le2 control-eval runs
workload_key = resco_config_eval_c_le2_completed_history
completed-active strict mapped count = 5
feasible profiles = 1
unit rule = one completed RESCO config run

completed-history singleton = Nature emissions real-road demand extraction
workload_key = nature_emissions_extract_c_le2_completed_history
completed-active strict mapped count = 1
feasible profiles = 1
unit rule = one completed extraction job

completed-history singleton = Nature emissions direct SUMO binary run
workload_key = nature_emissions_sumo_c_le2_completed_history
completed-active strict mapped count = 1
feasible profiles = 1
unit rule = one completed SUMO binary run

completed-history slice = Transit/FreqHRL c3_8 native-promotion persistent stress
workload_key = transit_native_promotion_c3_8_persistent_stress_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = Transit/FreqHRL c3_8 native real-demand batch validation
workload_key = transit_native_real_demand_batch_c3_8_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed native-control episode units

completed-history slice = Transit/FreqHRL c3_8 native real-demand alighting shards
workload_key = transit_native_real_demand_alighting_c3_8_completed_history
completed-active strict mapped count = 7
feasible profiles = 1
unit rule = parsed native-control episode units
```

Not yet closed:

```text
completed_active_production records = 3058
strict completed-active mapped count = 1658
representative completed-active mapped count = 2616
measurement_required = 442
cpu_sumo_transit_eval_or_control remaining = 73 / 3058
```

Module73 changes the production-population boundary, not the service map:
external `auto-adopted` stdin/wait-for processes with no scheduler id, no
scheduler log, and no reproducible progress unit are excluded from the
controlled-arrival theorem population.  They remain operational telemetry, but
including them in lambda would make the theorem claim arbitrary external
processes as schedulable workload.

The previous `freqduet_cpu_ablation|c_17_32` residual is now closed at the
command-shape level.  Module56 certifies records that actually invoke
`run_freqduet_ablation.py`; Module63 certifies explicit seed-index native
promotion validation records; Module74 certifies the remaining direct
`runner_v3.py`, paper-longtrain shell-wrapper, and parseable native residual
records separately.

The previous `freqduet_cpu_ablation|c_9_16` residual is now closed at the
command-shape level.  Module57 certifies only the direct `runner_v3.py` exact
config `configs_freqduet/F_allfreq_alllayers_hiro.yaml`, which accounts for
one completed/active production record.  Module58 certifies 110 c9_16
`run_freqduet_ablation.py` records with parseable units.  Module71 closes the
remaining completed-active c9_16 residual by splitting native-promotion records
into the slow bounded-wait stress profile and faster residual profile, and by
certifying residual `runner_v3.py` records separately.  A single coarse native
c9_16 class was rejected because its slow lower-service rate made the mapped
production-load LP infeasible.

The current `freqduet_cpu_ablation|c_3_8` group is also not fully closed.
Module60 certifies only parseable `run_freqduet_ablation.py` records using a
completed-history profile-1 lower-service point.  Module66 certifies the direct
`runner_v3.py` c3_8 records with explicit episodes.  The remaining 18 c3_8
records are native/control command shapes and stay in the regenerated Module53
manifest.

The current `freqduet_cpu_ablation|c_33_64` group is not fully closed either.
Module61 certifies only parseable `run_freqduet_ablation.py` records using a
completed-history profile-1 lower-service point.  Module68 certifies parseable
no-GPU native-promotion c33_64 records by splitting batch seed-episode work from
single-seed smoke/fix work.  The remaining 15 c33_64 records are direct-runner,
single-command-shape, or otherwise unparseable residuals and stay in the
Module53 manifest.

The current `freqduet_cpu_ablation|c_le2` group is almost fully closed.  Module62
certifies the direct `runner_v3.py` part with explicit episode counts.  Module76
certifies c_le2 `run_freqduet_ablation.py`, `run_baseline_rule.py`, preflight,
Transit/FreqHRL analysis-matrix, and Transit/FreqHRL merge command shapes.  The
only residual c_le2 records in this sub-bucket are two
`freqduet_autoadopt_spin.py` helpers, which are left unmeasured because their
stable progress unit is not part of the scheduler-controlled workload model.

The previous `bamor_cpu_training|c_le2` first-probe blocker is now closed at the
script-shape level.  Module77 certifies `train_compare_baselines.py`,
`train_bamor_mujoco.py`, and `run_bamor_diagnostic_shard.py` separately.  This is
not a pure CPU benchmark claim: several completed records specify `--device cuda`
while the scheduler estimated zero VRAM, so the safe interpretation is a
production-history completed-service certificate for those command shapes.

The previous `sumo_eval_cpu|c_le2` first-probe blocker is now closed at the
command-shape level.  Module78 certifies offline-sumo eval scripts,
H2Oplus/SimpleSAC complex shell eval jobs, ZSW metrics parsing, RESCO
config/main.py runs, and Nature-emissions extraction/SUMO singletons separately.
The service unit is one completed production command, which avoids overstating
work for resume, `--skip_existing`, checkpoint-wait, and shell-loop commands.

The previous `freqduet_cpu_ablation|c_3_8` first-probe blocker is now closed at
the native-service level.  Module79 identifies the residual rows as
Transit/FreqHRL native commands, not FreqDuet ablation commands, and certifies
native promotion, native real-demand batch, and alighting-shard service classes
separately.

The previous `freqduet_cpu_ablation|c_65p` top bucket is now closed at the
command-shape level.  Module69 splits it into high-CPU
`run_freqduet_ablation.py`, promoted ep100 shell-batch, and native-promotion
validation classes.  This does not authorize charging lower-CPU native
validation or non-ep100 shell batches to the c65p rates.

The remaining list can look larger after a module because broad residual
buckets are being split into theorem-facing command shapes.  The total
CPU/SUMO/transit measurement-required count is still lower after the latest
slices and population-boundary correction: 862 / 2471 after Module62, 803 /
2487 after Module63, 704 / 2553 after Module64, 659 / 2589 after Module65,
576 / 2679 after Module66/67, 532 / 2755 after Module68, 471 / 2781 after
Module69, 409 / 2797 after Module70, 350 / 2805 after Module71,
320 / 2840 after Module72, 258 / 2766 after Module73,
212 / 2766 after Module74, 169 / 2766 after Module75,
139 / 2766 after Module76, 113 / 2766 after Module77,
94 / 2910 after Module78, and 73 / 3058 after Module79.

Module67 refines the BAMOR c_3_8 CPU-training slice into script-level
certificates for train-compare, Mujoco, and diagnostic-shard commands.  Module75
does the same for BAMOR c_9_16, and Module77 does the same for BAMOR c_le2.
Remaining BAMOR c_17_32 records are still separate CPU/SUMO/transit obligations
until they receive their own service certificates.

Module65 closes only the ZSW TSP/SUMO c_le2 runner slice.  Module72 and Module78
then close the remaining CFCMT/offline-sumo/H2Oplus/RESCO/Nature c_le2 SUMO/eval
command shapes that have completed-history certificates.  Higher-CPU SUMO/eval
records remain separate obligations.

Module68 closes only parseable c33_64 native-promotion seed-unit records.  It
does not authorize charging c65p/c17_32 native-promotion records, direct
`runner_v3.py` records, or shell-expanded seed lists to the c33_64 batch service
rate.

Module69 closes the c65p high-CPU residual, but only through three explicit
parsers.  The shell command-substitution parser proves Python seed ranges by
AST without executing the production command.

Module70 closes the low-CPU Transit/FreqHRL validation/control residual by
splitting it into six explicit script-semantics classes.  It also corrects a few
low-CPU native Transit records that the broad keyword manifest had grouped under
`freqduet_cpu_ablation|c_le2`.  This does not authorize merging trading sweep,
policy training, surrogate validation, native simulator validation, and
import-smoke work into one service class.

Module71 closes the previous c9_16 residual command-shape blocker with three
explicit certificates.  The split is required by the capacity theorem: the
bounded-wait stress profile has the slowest lower-service rate and cannot be
used as the service class for all native-promotion c9_16 work without making
the mapped load infeasible.

Module72 closes the CFCMT portion of `sumo_eval_cpu|c_le2` with six explicit
script-semantics certificates.  It does not authorize mapping offline-sumo,
H2Oplus, RESCO/config, ZSW metrics-parser, or Nature-emissions SUMO records to
the CFCMT service classes.

Module73 removes the former `transit_misc_cpu|c_le2` first-probe bucket by
tightening the theorem population rather than by pretending those external
stdin/wait-for processes have a measured Transit service curve.

Module74 closes the former `freqduet_cpu_ablation|c_17_32` first-probe bucket
with three explicit completed-history service classes: direct runner_v3
episodes, paper-longtrain completed shards, and residual native seed-episodes.
The longtrain class is intentionally shard-based because its production command
uses `--skip-existing`.

Module75 closes the former `bamor_cpu_training|c_9_16` first-probe blocker with
three explicit completed-history service classes: train-compare, Mujoco, and
diagnostic-shard training-step work.  The split is required because those three
script shapes have materially different service rates.

Module76 closes most of the former `freqduet_cpu_ablation|c_le2` residual with
five explicit completed-history service classes: FreqDuet ablation episodes,
baseline-rule episodes, preflight checks, Transit/FreqHRL analysis jobs, and
Transit/FreqHRL merge jobs.  It does not claim the two auto-adopt spin helpers.

Module77 closes the former `bamor_cpu_training|c_le2` first-probe blocker with
three explicit completed-history service classes: train-compare, Mujoco, and
diagnostic-shard training-step work.  The certificate keeps CUDA-flagged
zero-estimated-VRAM records in production-history semantics instead of using them
as generic CPU-only curves.

Module78 closes the former `sumo_eval_cpu|c_le2` first-probe blocker with six
explicit completed-history service classes.  It intentionally uses completed
production-command units rather than parsed episode/checkpoint counts, making the
new mapped-capacity slack smaller but more defensible.

Module79 closes the former `freqduet_cpu_ablation|c_3_8` first-probe blocker
with three explicit native-service classes.  This corrects the manifest's broad
keyword bucket rather than pretending the records are FreqDuet ablations.

## Interpretation

The mapped capacity slack is positive, but that proves only that the already
measured and mapped production slice lies inside the measured capacity region.
It does not prove that the full production load is stabilizable.

The mapped-capacity slack is now much tighter than before Module59 because the
SimpleSAC and Module74 paper-longtrain slices use conservative completed-history
profile-1 lower-service rates.  This tightening is part of the theorem
condition audit, not a reason to relabel slow tasks:

```text
strict mapped delta = 0.000008435
```

This is a theorem-condition warning, not a reason to relabel unmeasured tasks.
Either future work measures higher SimpleSAC co-location profiles, or this slice
stays as a conservative low-throughput service class.

Representative mappings are diagnostic.  They are not theorem-grade unless the
bucket has either a measured service curve or a separate equivalence certificate.

The live scheduler oracle trace / lower-service bridge is also not globally
closed until real production candidate traces with lower-service semantics are
collected and audited.

## Remaining Probe Order

The next production CPU/SUMO/transit slices should be attacked in this order:

```text
bamor_cpu_training|c_17_32
freqduet_cpu_ablation|c_33_64 residual direct-runner/single-command shapes
sumo_eval_cpu|c_3_8
transit_freqhrl_cpu_validation|c_3_8
transit_freqhrl_cpu_validation|c_17_32
sumo_eval_cpu|c_33_64
transit_freqhrl_cpu_validation|c_33_64
```

For every slice, the required closure pattern is:

```text
1. define a command-shape-specific strict classifier;
2. run a progress-bearing service curve over profiles 1,2,4,8 or until capacity boundary;
3. store profile summaries in md/experiment_artifacts;
4. add the measured feasible profiles and boundary to ServiceRateCache;
5. add or update the corresponding TaskSet;
6. rerun production load, coverage, and probe manifest artifacts;
7. run calibration tests and push only after validation passes.
```
