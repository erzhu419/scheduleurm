# Module53 CPU/SUMO/Transit Probe Manifest

Date: 2026-06-09

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

## Status After Module72

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

## Current Remaining Bucket

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 320
cpu_cores median = 4
cpu_cores p90 = 32
cpu_cores max = 61
ram_mb median = 4096.000
ram_mb p90 = 65536
ram_mb max = 131072
theorem_status = measurement_required
```

## Top Remaining Sub-Buckets

| Sub-Bucket | Count | Fraction |
|---|---:|---:|
| `transit_misc_cpu|c_le2` | 48 | 0.150000 |
| `freqduet_cpu_ablation|c_17_32` | 46 | 0.143750 |
| `bamor_cpu_training|c_9_16` | 43 | 0.134375 |
| `freqduet_cpu_ablation|c_le2` | 36 | 0.112500 |
| `bamor_cpu_training|c_le2` | 28 | 0.087500 |
| `sumo_eval_cpu|c_le2` | 21 | 0.065625 |
| `freqduet_cpu_ablation|c_3_8` | 18 | 0.056250 |
| `bamor_cpu_training|c_17_32` | 17 | 0.053125 |
| `freqduet_cpu_ablation|c_33_64` | 15 | 0.046875 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.040625 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.037500 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.018750 |

## Next Probe Order

The regenerated manifest recommends this first pass over the remaining bucket:

```text
transit_misc_cpu|c_le2
freqduet_cpu_ablation|c_17_32 residual command shapes
bamor_cpu_training|c_9_16
freqduet_cpu_ablation|c_le2 residual command shapes
bamor_cpu_training|c_le2
sumo_eval_cpu|c_le2 residual command shapes
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
records into six script-level completed-history service classes.  The global
theorem remains open because 320
completed/active production records in the CPU/SUMO/transit family still
require measured curves or equivalence certificates, led by 48
`transit_misc_cpu|c_le2` records, 46 residual
`freqduet_cpu_ablation|c_17_32` records, 43 `bamor_cpu_training|c_9_16`
records, 36 residual `freqduet_cpu_ablation|c_le2` records, and 28
`bamor_cpu_training|c_le2` records.
