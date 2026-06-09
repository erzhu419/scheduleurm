# OR Submission Gap Closure Plan

Date: 2026-06-10

This note records the remaining gaps before the Scheduleurm line can be written
as an OR/Stochastic Systems paper. It is intentionally stricter than an
engineering experiment checklist: each item states what a reviewer can ask, what
must be shown, and what artifact should answer it.

## Current Assessment

The mathematical spine is close to OR-grade:

```text
finite-feature fabric-cover candidate approximation
-> robust candidate MaxWeight with explicit slack accounting
-> bounded conditional second moment / finite-set Foster recurrence
-> approximate-oracle and statewise feasible-family variants
```

The main theorem should be the approximate-oracle calibrated version:

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle
```

For the actual scheduler with dynamic feasible families, the stronger aligned
statement is:

```text
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

The paper is not yet submission-ready because the remaining work is mostly
artifact, calibration, and empirical closure, not abstract theorem invention.

## Gap 1: Theorem-Condition Calibration

Reviewer question:

```text
Your theorem needs δ > Lρ + ε_est + β + α1. Do your measured Scheduleurm data
actually satisfy this inequality?
```

Required output:

```text
L, rho, Lrho
epsilon_est
P0, beta
alpha0, alpha1
B
delta
eta = delta - (Lrho + epsilon_est + beta + alpha1)
finite-set threshold N from B + P0 + alpha0
```

Required artifact:

```text
algorithm/experiments/slack_accounting.py
algorithm/experiments/empirical_slack_certificate.py
algorithm/experiments/production_load_certificate.py
algorithm/experiments/production_coverage_drilldown.py
algorithm/experiments/production_bucket_probe_manifest.py
algorithm/oracle_trace.py
algorithm/experiments/scheduler_oracle_trace_audit.py
algorithm/experiments/theorem_oracle_trace_bridge.py
algorithm/experiments/production_cpu_workload_curve.py
algorithm/experiments/oracle_trace_enrichment.py
md/experiment_module30_slack_accounting.md
md/experiment_module48_theorem_condition_calibration.md
md/experiment_module49_production_load_capacity.md
md/experiment_module51_production_coverage_drilldown.md
md/experiment_module53_cpu_sumo_transit_probe_manifest.md
md/experiment_module50_scheduler_oracle_trace.md
md/experiment_module52_theorem_oracle_trace_bridge.md
md/experiment_module54_production_cpu_workload_curve_runner.md
md/experiment_module55_oracle_trace_lower_service_enrichment.md
md/experiment_module57_freqduet_runner_v3_c9_16_curve.md
md/experiment_module58_freqduet_ablation_c9_16_curve.md
md/experiment_module59_simple_sac_sumo_eval_cle2_completed_history.md
md/experiment_module60_freqduet_ablation_c3_8_completed_history.md
md/experiment_module61_freqduet_ablation_c33_64_completed_history.md
md/experiment_module62_freqduet_runner_v3_c_le2_completed_history.md
md/experiment_module63_transit_native_promotion_c17_32_seedrange_completed_history.md
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
md/experiment_artifacts/module48_portfolio_slack_certificate.json
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module53_cpu_sumo_transit_probe_manifest.json
md/experiment_artifacts/module50_scheduler_oracle_trace_status.json
md/experiment_artifacts/module52_theorem_oracle_trace_bridge.json
md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.json
md/experiment_artifacts/module55_oracle_trace_enrichment_status.json
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124.json
md/experiment_artifacts/module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8.json
md/experiment_artifacts/module57_freqduet_runner_v3_c9_16_jtl110cpu2_curve_p1248.json
md/experiment_artifacts/module58_freqduet_ablation_c9_16_jtl110cpu2_curve_p1248.json
md/experiment_artifacts/module58_freqduet_ablation_c9_16_completed_wallclock_audit.json
md/experiment_artifacts/module59_simple_sac_sumo_eval_cle2_completed_history.json
md/experiment_artifacts/module60_freqduet_ablation_c3_8_completed_history.json
md/experiment_artifacts/module61_freqduet_ablation_c33_64_completed_history.json
md/experiment_artifacts/module62_freqduet_runner_v3_c_le2_completed_history.json
md/experiment_artifacts/module63_transit_native_promotion_c17_32_seedrange_completed_history.json
md/experiment_artifacts/module65_zsw_tsp_sumo_eval_c_le2_completed_history.json
md/experiment_artifacts/module66_freqduet_runner_v3_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_train_compare_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_mujoco_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_diagnostic_shard_c3_8_completed_history.json
md/experiment_artifacts/module68_transit_native_promotion_c33_64_batch_completed_history.json
md/experiment_artifacts/module68_transit_native_promotion_c33_64_single_seed_completed_history.json
md/experiment_artifacts/module69_freqduet_ablation_c65p_completed_history.json
md/experiment_artifacts/module69_freqduet_promoted_ep100_c65p_completed_history.json
md/experiment_artifacts/module69_transit_native_promotion_c65p_completed_history.json
md/experiment_artifacts/module70_transit_trading_sweep_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_trading_policy_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_surrogate_validation_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_native_promotion_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_native_control_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_freqhrl_import_smoke_c_le2_completed_history.json
md/experiment_artifacts/module71_transit_native_promotion_c9_16_bounded_wait_completed_history.json
md/experiment_artifacts/module71_transit_native_promotion_c9_16_residual_completed_history.json
md/experiment_artifacts/module71_freqduet_runner_v3_c9_16_residual_completed_history.json
md/experiment_artifacts/module72_cfcmt_feed_conversion_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_env_validation_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_sumo_generation_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_snapshot_generation_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_policy_rollout_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_traffic_signal_phase2_c_le2_completed_history.json
```

Status:

```text
Infrastructure exists for separate components:
  fabric_metric.py        -> L, rho
  penalty_fit.py          -> P0, beta
  oracle_audit.py         -> alpha0, alpha1
  capacity_lp.py          -> delta, eta

Module48 now provides one consolidated positive pass/fail certificate for the
measured hybrid_research_portfolio finite service-action slice:
  delta = 0.066921842
  eta   = 0.066921842
  B     = 6361.354514
  N     = 95072
  Lrho = epsilon_est = beta = alpha1 = 0 for the exact finite-slice certificate.

Remaining scope limitation:
  Module49 estimates production load from a 30-day Scheduleurm history window.
  The mapped measured-bucket capacity LP is positive:
    strict measured mapping delta = 0.000050804
    representative mapping delta = 0.000050804
  Full global theorem coverage is still open because the representative run
  leaves 1602 / 5980 tasks unmapped and maps 1896 tasks only by representative
  bucket equivalence rather than theorem-grade service measurement.

  Module51 tightens the population definition.  Module73 further excludes
  unobservable external auto-adopted stdin/wait-for processes from the
  controlled-arrival theorem population when they have no scheduler id, no
  scheduler log, and no reproducible progress unit.  In the reviewer-facing
  completed-active production view, representative coverage is:
    records = 2766
    mapped = 2196
    strict mapped = 1288
    measurement_required = 570
    mapped_fraction = 0.793926
  The dominant remaining bucket is:
    cpu_sumo_transit_eval_or_control = 212 / 2766 records
  Therefore the next production-coverage closure target is a theorem-grade
  service curve for the CPU/SUMO/transit evaluation/control family, not another
  generic q01/q11 GPU RL probe.

  Module53 turns that target into concrete sub-buckets.  Module56 closed the
  exact run_freqduet_ablation.py slice inside the previous top sub-bucket:
    run_freqduet_ablation.py within freqduet_cpu_ablation|c_17_32 -> freqduet_cpu_ablation_c17_32
    feasible profiles = 1,2,4
    capacity boundary = 8
    completed-active strict mapped count = 147
    residual freqduet_cpu_ablation|c_17_32 needing measurement after Module63 = 46

  Module57 closes one exact direct-runner config inside c9_16:
    runner_v3.py --config configs_freqduet/F_allfreq_alllayers_hiro.yaml
      -> freqduet_runner_v3_allfreq_alllayers_c9_16
    feasible profiles = 1,2,4,8
    completed-active strict mapped count = 1
  This is deliberately narrow; it does not close the broader c9_16 ablation
  family.

  Module58 closes the broader c9_16 run_freqduet_ablation.py command-shape slice:
    run_freqduet_ablation.py within freqduet_cpu_ablation|c_9_16
      -> freqduet_cpu_ablation_c9_16
    feasible profiles = 1,2,4,8
    completed-active strict mapped count = 110
    parsed completed-active work = 138900 episode units
    min completed wall-clock rate = 0.289775 episode/s
  Module71 closes the remaining completed-active c9_16 residual command shapes:
    bounded-wait native_promotion_replan_validation
      -> transit_native_promotion_c9_16_bounded_wait_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 7
    parsed completed-history work = 2192 seed-episode units
    min completed wall-clock rate = 0.003792 seed-episode/s

    residual native_promotion_replan_validation
      -> transit_native_promotion_c9_16_residual_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 40
    parsed completed-history work = 7723 seed-episode units
    min completed wall-clock rate = 0.035085 seed-episode/s

    residual runner_v3.py within freqduet_cpu_ablation|c_9_16
      -> freqduet_runner_v3_c9_16_residual_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 12
    parsed completed-history work = 440 episode units
    min completed wall-clock rate = 0.030460 episode/s
  The coarse one-class native c9_16 certificate was rejected because it made the
  mapped production-load LP infeasible; the final split is a theorem-condition
  correction, not a cosmetic relabeling.

  Module59 closes the clean SimpleSAC c_le2 SUMO eval sub-slice:
    clean run_multiseed_eval.sh within sumo_eval_cpu|c_le2
      -> sumo_eval_simple_sac_c_le2
    feasible profiles = 1
    completed-active strict mapped count = 54
    min completed wall-clock rate = 0.001525164 eval/s
  This improves coverage but makes mapped-capacity slack tight:
    strict mapped delta after Module74 raw-window LP = 0.000050804

  Module72 closes the CFCMT portion of `sumo_eval_cpu|c_le2` with six
  script-level completed-history certificates:
    gtfs_to_h2o_xlsx.py / lta_to_h2o_xlsx.py
      -> cfcmt_feed_conversion_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 5
    parsed completed-history work = 5 feed-conversion jobs
    min completed wall-clock rate = 0.000677 feed-conversion/s

    validate_h2o_city_env.py
      -> cfcmt_env_validation_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 5
    parsed completed-history work = 87200 validation-step units
    min completed wall-clock rate = 11.792971 validation-step/s

    sumo_apc_avl_sumo_generation
      -> cfcmt_sumo_generation_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 4
    parsed completed-history work = 106200 simulated-second units
    min completed wall-clock rate = 18.527091 simulated-second/s

    sumo_apc_avl_snapshot_generation
      -> cfcmt_snapshot_generation_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 5
    parsed completed-history work = 1560 snapshot-window units
    min completed wall-clock rate = 0.028279 snapshot-window/s

    sumo_policy_rollout_validation.py
      -> cfcmt_policy_rollout_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 10
    parsed completed-history work = 140820 policy-event-budget units
    min completed wall-clock rate = 10.848109 policy-event-budget/s

    traffic_signal_sumo_phase2.py
      -> cfcmt_traffic_signal_phase2_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 1
    parsed completed-history work = 1 phase2-run unit
    min completed wall-clock rate = 0.001202 phase2-run/s
  Policy-rollout and snapshot-generation commands are matched before
  SUMO-generation commands because their stage2-report paths contain
  `sumo_generation`; this prevents path strings from collapsing distinct
  service units into one class.  Module72 does not claim offline-sumo, H2Oplus,
  RESCO/config, ZSW metrics-parser, or Nature-emissions SUMO records.

  Module60 closes the parseable c3_8 FreqDuet ablation sub-slice:
    run_freqduet_ablation.py within freqduet_cpu_ablation|c_3_8
      -> freqduet_cpu_ablation_c3_8_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 40
    parsed completed-active work = 15173 episode units
    min completed wall-clock rate = 0.011071601 episode/s
  This improves coverage but, before Module66, did not close the residual c3_8
  runner/native command shapes.

  Module61 closes the parseable c33_64 FreqDuet ablation sub-slice:
    run_freqduet_ablation.py within freqduet_cpu_ablation|c_33_64
      -> freqduet_cpu_ablation_c33_64_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 63
    parsed completed-active work = 156140 episode units
    min completed wall-clock rate = 0.391511591 episode/s
  This improves coverage but leaves native/direct-runner c33_64 residuals.

  Module62 closes the direct runner_v3 c_le2 FreqDuet sub-slice:
    runner_v3.py within freqduet_cpu_ablation|c_le2
      -> freqduet_runner_v3_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 84
    parsed completed-active work = 4661 episode units
    min completed wall-clock rate = 0.010924045 episode/s
  This improves coverage but leaves c_le2 ablation, baseline-rule, scheduler
  wait, native-validation, and shell-loop residuals.

  Module63 closes the seed-range Transit native-promotion c17_32 sub-slice:
    native_promotion_replan_validation with explicit seed-index range
      -> transit_native_promotion_c17_32_seedrange_completed_history
    feasible profiles = 1
    completed-history service records used = 71
    current completed-active strict mapped count = 91
    parsed completed-history work = 17760 seed-episode units
    min completed wall-clock rate = 0.075456 seed-episode/s
  This improves coverage but leaves direct-runner and non-seed-range native
  c17_32 command shapes.

  Module64 closes the BAMOR c3_8 CPU-training sub-slice:
    train_compare_baselines.py / train_bamor_mujoco.py / run_bamor_diagnostic_shard.py
      with parseable training-step units -> bamor_cpu_training_c3_8_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 142
    parsed completed-history work = 25252000 training-step units
    min completed wall-clock rate = 10.055276 training-step/s
  This improves coverage but leaves BAMOR c_le2/c9_16/c17_32 records as separate
  obligations.

  Module67 refines the BAMOR c3_8 CPU-training sub-slice into script-level
  lower-service classes:
    train_compare_baselines.py -> bamor_train_compare_c3_8_completed_history
      completed-active strict mapped count = 58
      min completed wall-clock rate = 10.055276 training-step/s
    train_bamor_mujoco.py -> bamor_mujoco_c3_8_completed_history
      completed-active strict mapped count = 123
      min completed wall-clock rate = 37.510192 training-step/s
    run_bamor_diagnostic_shard.py -> bamor_diagnostic_shard_c3_8_completed_history
      completed-active strict mapped count = 25
      min completed wall-clock rate = 412.612802 training-step/s
  This refinement is required for the capacity theorem: the latest raw-window
  BAMOR load exceeds the broad Module64 class if every BAMOR task is charged
  the slowest train-compare lower-service rate.

  Module65 closes the ZSW TSP/SUMO c_le2 runner sub-slice:
    baseline_runner.py / m21_cycle_conserving_tsp_runner.py /
      oracle_tsp_runner.py / m2_scored_oracle_tsp_runner.py with explicit --duration
      -> zsw_tsp_sumo_eval_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 50
    parsed completed-history work = 900000 simulated-second units
    min completed wall-clock rate = 5.896139 simulated-second/s
  This improves coverage but leaves CFCMT/offline-sumo/H2Oplus/direct-SUMO
  c_le2 records as separate obligations.

  Module66 closes the direct runner_v3 c3_8 FreqDuet sub-slice:
    runner_v3.py within freqduet_cpu_ablation|c_3_8
      -> freqduet_runner_v3_c3_8_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 86
    parsed completed-history work = 2709 episode units
    min completed wall-clock rate = 0.006679 episode/s
  This improves coverage but leaves native/control c3_8 residuals.

  Module68 closes the c33_64 Transit native-promotion seed-unit sub-slices:
    native_promotion_replan_validation batch records with parseable seed work
      -> transit_native_promotion_c33_64_batch_completed_history
    feasible profiles = 1
    completed-history service records used = 45
    current completed-active strict mapped count = 55
    parsed completed-history work = 5922 seed-episode units
    min completed wall-clock rate = 0.060204 seed-episode/s

    native_promotion_replan_validation single-seed smoke/fix records
      -> transit_native_promotion_c33_64_single_seed_completed_history
    feasible profiles = 1
    certificate record count = 2
    parsed completed-history work = 2 seed-episode units
    min completed wall-clock rate = 0.003368 seed-episode/s
  This improves coverage and prevents single-seed smoke/fix records from
  depressing the batch lower-service class.  It leaves direct-runner,
  single-command-shape, and unparseable c33_64 residuals in Module53.

  Module69 closes the c65p high-CPU residual with three command-shape classes:
    run_freqduet_ablation.py within freqduet_cpu_ablation|c_65p
      -> freqduet_cpu_ablation_c65p_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 7
    parsed completed-history work = 18430 episode units
    min completed wall-clock rate = 1.099736 episode/s

    run_freqduet_promoted_ep100_hpc_batch.sh within c_65p
      -> freqduet_promoted_ep100_c65p_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 6
    parsed completed-history work = 60000 episode units
    min completed wall-clock rate = 1.727479 episode/s

    native_promotion_replan_validation within c_65p
      -> transit_native_promotion_c65p_completed_history
    feasible profiles = 1
    completed-history service records used = 48
    current completed-active strict mapped count after Module71 = 49
    parsed completed-history work = 7026 seed-episode units over done records
    min completed wall-clock rate = 0.182131 seed-episode/s
  The native parser proves shell-generated Python seed ranges by AST without
  executing the command.  This removes the previous c65p top residual from the
  regenerated Module53 manifest.

  Module70 closes the former top low-CPU Transit/FreqHRL validation/control
  residual with six script-semantics classes:
    trading sweep/performance/pressure/encoder validation
      -> transit_trading_sweep_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 14
    parsed completed-history work = 42238800 market-step units
    min completed wall-clock rate = 899.891032 market-step/s

    trading policy entry / PPO actor-critic validation
      -> transit_trading_policy_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 18
    parsed completed-history work = 4147200 policy-market-step units
    min completed wall-clock rate = 650.109326 policy-market-step/s

    surrogate validation / PPO surrogate validation
      -> transit_surrogate_validation_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 3
    parsed completed-history work = 49344 surrogate-corridor-step units
    min completed wall-clock rate = 84.110249 surrogate-corridor-step/s

    native_promotion_replan_validation within c_le2
      -> transit_native_promotion_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 19
    parsed completed-history work = 522 native-variant-episode units
    min completed wall-clock rate = 0.005407 native-variant-episode/s

    native wait-credit / real-demand control validation
      -> transit_native_control_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 7
    parsed completed-history work = 148 native-control-episode units
    min completed wall-clock rate = 0.028357 native-control-episode/s

    Windows import smoke checks
      -> transit_freqhrl_import_smoke_c_le2_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 1
    parsed completed-history work = 1 import-check unit
    min completed wall-clock rate = 0.001756 import-check/s
  This removes `transit_freqhrl_cpu_validation|c_le2` from the regenerated
  Module53 first-probe order.  It deliberately does not merge trading,
  surrogate, native simulator validation, native control, and import-smoke work
  into one service class.

  Module71 closes the previous c9_16 residual first-probe blocker with the
  three completed-history certificates listed above.  This removes
  `freqduet_cpu_ablation|c_9_16` from the regenerated Module53 first-probe
  order while preserving the slow bounded-wait profile as a separate
  lower-service class.

  Module72 closes the CFCMT portion of `sumo_eval_cpu|c_le2`, but not the
  remaining offline-sumo/H2Oplus/RESCO/ZSW/Nature SUMO rows.

  Module73 removes the former `transit_misc_cpu|c_le2` first-probe blocker from
  the controlled-arrival theorem population.  Those records are external
  auto-adopted stdin/wait-for processes with no scheduler id, no scheduler log,
  and no reproducible progress unit.  They remain operational telemetry, but
  they cannot be charged to a Scheduleurm service curve without making the
  theorem claim arbitrary external processes.

  Module74 closes the previous `freqduet_cpu_ablation|c_17_32` first-probe
  blocker with three completed-history certificates:
    direct runner_v3.py c17_32
      -> freqduet_runner_v3_c17_32_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 16
    parsed completed-history work = 560 episode units
    min completed wall-clock rate = 0.017634 episode/s

    run_freqduet_paper_longtrain_matrix.sh c17_32
      -> freqduet_paper_longtrain_c17_32_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 16
    parsed completed-history work = 16 longtrain-shard units
    min completed wall-clock rate = 0.000064 longtrain-shard/s
    note = uses shard completion because --skip-existing prevents safe episode-count accounting

    residual native_promotion_replan_validation c17_32
      -> transit_native_promotion_c17_32_residual_completed_history
    feasible profiles = 1
    completed-active strict mapped count = 14
    parsed completed-history work = 970 seed-episode units
    min completed wall-clock rate = 0.059097 seed-episode/s

  The current remaining top probe order is:
    bamor_cpu_training|c_9_16
    freqduet_cpu_ablation|c_le2 residual command shapes
    bamor_cpu_training|c_le2
    sumo_eval_cpu|c_le2 residual command shapes
    freqduet_cpu_ablation|c_3_8 residual native/control command shapes
    bamor_cpu_training|c_17_32
    freqduet_cpu_ablation|c_33_64 residual direct-runner/single-command shapes

  Module54 adds the actual progress-bearing CPU-only runner used for the first
  theorem-grade production slice.  Module56 validates it on jtl110cpu2:
    freqduet_cpu_ablation|c_17_32
    profile 1 aggregate = 0.447090 episode/s
    profile 2 aggregate = 0.539602 episode/s
    profile 4 aggregate = 0.878374 episode/s
    profile 8 = capacity boundary
    cpu/task = 24
    ram/task = 65536
    work_items/task = 24
    theorem_status = measured_sub_bucket_loaded_into_service_cache
  The remaining empirical step is to repeat this for the next sub-buckets and
  rerun Modules49/51/slack accounting after each new measured slice.

  Module50 adds the missing live scheduler candidate-set trace hook.  It is
  disabled by default and records the actual candidate family used by
  pick_placement when SCHEDULEURM_ORACLE_AUDIT_LOG is set.  Unit validation
  proves the selected action is scheduler-score best on a synthetic two-GPU
  decision.  Current production status is NO_TRACE, so the live alpha0/alpha1
  theorem certificate remains open until real candidate-family traces with
  robust lower-service semantics are collected.

  Module52 adds the theorem-side gate.  It refuses scheduler-sort-key-only
  traces and accepts only slots with:
    score_semantics = robust_maxweight_lower_service
    queue_vector
    lower_service
    penalty_units
  Current status is NO_TRACE, but the conversion path to oracle_audit.py is now
  implemented and unit-tested.

  Module55 adds the missing enrichment bridge from scheduler-sort-key trace to
  Module52 theorem trace:
    trace + measured lower-service lookup + queue_vector
      -> robust_maxweight_lower_service slots
      -> alpha0/alpha1 audit
  The current production status is still NO_TRACE because
  ~/.claude/scheduler/oracle_trace.jsonl does not exist.  The software path is
  closed; the remaining blocker is real trace collection plus complete measured
  lower-service coverage for every candidate in each traced slot.
```

## Gap 2: q10 Real CPU/Data-Loader Trace and q00 Control Bucket Closure

Reviewer question:

```text
Why is q10_cpu_host_bound based on a real workload measured under the same
benchmark discipline, and what bucket does that claim cover?
```

Required output:

```text
real CPU-heavy or data-loader-heavy progress-bearing workload;
profiles measured on a declared stable bucket;
legacy-comparable cap measured on that same bucket;
candidate profile measured or certified;
q10 taskset promoted only after those conditions hold.
```

Current status:

```text
modules25+33 have a real local CPU-heavy curve;
profiles 1-9 are measured;
profile 10 is a measured local capacity boundary;
active q10 taskset uses cpu_heavy_local_bench;
legacy cap is profile 9 on the same local bucket;
candidate profile is profile 8.
```

q00 status:

```text
modules23+24+38 have the exact local light-control curve;
profiles 1-13 are measured;
profile 14 is a measured local scheduling-capacity boundary;
active q00 taskset uses light_control_local;
legacy cap is profile 1 on the same local bucket;
candidate profile is profile 13.
```

Required artifact:

```text
md/experiment_module31_q10_real_cpu_trace.md
md/experiment_module38_q00_light_control_extended_curve.md
```

Remaining breadth item:

```text
Remote CPU-node or data-loader-heavy q10 replication is still useful, but it is
not required for the declared local CPU-bucket q10 comparison. Remote/light
control-plane replication is likewise useful for q00 generalization but not
required for the declared local q00 comparison.
```

## Gap 3: SOTA Wording

Reviewer question:

```text
Did you directly run Gavel/Pollux/Sia/IADeep, or are these stylized policy
baselines on the Scheduleurm service cache?
```

Required wording:

```text
We compare against SOTA-style replay baselines that reproduce policy semantics
on the same measured Scheduleurm service cache. We do not claim direct binary
execution of external schedulers unless a later experiment explicitly runs or
faithfully ports their full allocation semantics.
```

Current status:

```text
module29 has per-taskset SOTA comparisons and fixed-policy matrix;
no individual fixed SOTA-style policy Pareto-dominates Scheduleurm candidate.
```

## Gap 4: Replay-to-Live Validation

Reviewer question:

```text
Your replay result is plausible, but does it agree with small real live runs?
```

Required output:

```text
small q01 live run; completed by module39;
small q11 live run; completed by module46 after profile-10 robust boundary correction;
small portfolio live sanity run; completed by module47;
predicted all-job completion / mean-flow from replay;
observed live completion / progress-window JCT;
relative error and confidence interval;
explicit statement that long production jobs are not required to run to natural
completion for service-curve calibration.
```

Required artifact:

```text
algorithm/experiments/live_validation.py
algorithm/experiments/portfolio_live_proxy.py
md/experiment_module32_live_replay_sanity.md
md/experiment_module39_46_live_replay_sanity_and_q11_robust_boundary.md
md/experiment_module47_portfolio_live_sanity.md
```

Current status:

```text
q01 module39: usable_for_live_sanity = true
q11 module46: usable_for_live_sanity = true
portfolio module47: usable_for_live_sanity = true
portfolio replay-to-live relative errors:
  makespan  = 0.007349
  mean-flow = 0.005417
  p90-flow  = 0.007964
```

## Gap 5: Lean Artifact Repackaging

Reviewer question:

```text
Does the uploaded Lean artifact contain exactly the theorem names cited in the
paper, with a fresh build log and no sorry/admit/axiom?
```

Required output:

```text
fresh ScheduleurmUpload.lean
sha256 hash
git commit of proof repo
lake build Scheduleurm
lake env lean ScheduleurmUpload.lean
grep for sorry/admit/axiom
theorem-name search log
```

Required artifact:

```text
md/lean_verification_submission.md
```

Current status:

```text
completed on 2026-06-08;
proof commit = 23b101432067cc005512f7667810ec03b8cffb77;
ScheduleurmUpload.lean sha256 = af79be4416e4c4add0fe41663fc0927af7058fe04412908b6688a8409227f01b;
lake build Scheduleurm = PASS;
lake env lean ScheduleurmUpload.lean = PASS;
sorry/admit/axiom grep = clean;
submission theorem-name grep = all present.
```

## Gap 6: OR Related Work Framing

Reviewer question:

```text
Is this a scheduling-systems paper with some math, or a queueing/control paper
with a system-backed calibration?
```

Required framing:

```text
stochastic processing networks
MaxWeight / backpressure
capacity regions and support functions
approximate MaxWeight / restricted candidate controls
queueing networks with learning / unknown service
restless or regime-switching service extensions
```

Systems work such as Gavel/Pollux/Sia/IADeep should be used mainly for
experiment design and baseline semantics, not as the theoretical related-work
center.

Required artifact:

```text
md/or_related_work_outline.md
```

## Submission Readiness Standard

The paper can move from "research prototype" to "submission draft" only after:

```text
1. slack accounting table has eta > 0 or clearly explains why the current
   workload/load point is outside certified stability;
2. q10 and q00 are real for declared buckets, or explicitly removed from theorem-grade claims;
3. SOTA claims use the honest SOTA-style replay wording;
4. at least one small live validation connects replay to real completion/JCT;
5. Lean artifact is freshly repackaged and theorem names match the paper;
6. the main manuscript exists as a coherent OR paper, not a pile of md notes.
```
