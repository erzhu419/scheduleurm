# Production Global Closure Remaining

Date: 2026-06-09

This note records the current gap after Module68.  It should be read together
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
completed-active strict mapped count = 71
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = BAMOR c_3_8 train_compare_baselines.py
workload_key = bamor_train_compare_c3_8_completed_history
completed-active strict mapped count = 58
feasible profiles = 1
unit rule = parsed training steps

completed-history slice = BAMOR c_3_8 train_bamor_mujoco.py
workload_key = bamor_mujoco_c3_8_completed_history
completed-active strict mapped count = 89
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
completed-active strict mapped count = 45
feasible profiles = 1
unit rule = parsed seed-count times episodes

completed-history slice = Transit native_promotion_replan_validation single-seed smoke/fix within c_33_64
workload_key = transit_native_promotion_c33_64_single_seed_completed_history
certificate record count = 2
feasible profiles = 1
unit rule = parsed seed-count times episodes
```

Not yet closed:

```text
completed_active_production records = 2755
strict completed-active mapped count = 956
representative completed-active mapped count = 1867
measurement_required = 888
cpu_sumo_transit_eval_or_control remaining = 532 / 2755
```

The original `freqduet_cpu_ablation|c_17_32` group is not fully closed.  Module56
only certifies records that actually invoke `run_freqduet_ablation.py`.  The
Module63 seed-range certificate additionally closes a Transit native-promotion
validation sub-slice.  The remaining `freqduet_cpu_ablation|c_17_32` records
include 46 completed/active tasks with different command shapes, such as direct
`runner_v3.py` invocations or native commands without parseable seed-index
ranges, so neither the Module56 curve nor the Module63 lower-service point may
be used to certify them.

The current `freqduet_cpu_ablation|c_9_16` group is still not fully closed.
Module57 certifies only the direct `runner_v3.py` exact config
`configs_freqduet/F_allfreq_alllayers_hiro.yaml`, which accounts for one
completed/active production record.  Module58 certifies 110 c9_16
`run_freqduet_ablation.py` records with parseable units.  The remaining c9_16
records have different command shapes and stay in the Module53 manifest.

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
single-seed smoke/fix work.  The remaining 17 c33_64 records are direct-runner,
single-command-shape, or otherwise unparseable residuals and stay in the
Module53 manifest.

The current `freqduet_cpu_ablation|c_le2` group is not fully closed.  Module62
certifies the direct `runner_v3.py` part with explicit episode counts.  The
remaining c_le2 records are ablation, baseline-rule, scheduler wait, native
validation, or shell-loop command shapes.

The remaining list can look larger after a module because broad residual
buckets are being split into theorem-facing command shapes.  The total
CPU/SUMO/transit measurement-required count is still lower after the latest
slices: 862 / 2471 after Module62, 803 / 2487 after Module63, 704 / 2553 after
Module64, 659 / 2589 after Module65, 576 / 2679 after Module66/67, and
532 / 2755 after Module68.

Module67 refines the BAMOR c_3_8 CPU-training slice into script-level
certificates for train-compare, Mujoco, and diagnostic-shard commands.  Remaining
BAMOR c_le2, c_9_16, and c_17_32 records are still separate CPU/SUMO/transit
obligations until they receive their own service certificates.

Module65 closes only the ZSW TSP/SUMO c_le2 runner slice.  Remaining
`sumo_eval_cpu|c_le2` records are CFCMT/offline-sumo/H2Oplus/direct-SUMO
command shapes and remain separate obligations.

Module68 closes only parseable c33_64 native-promotion seed-unit records.  It
does not authorize charging c65p/c17_32 native-promotion records, direct
`runner_v3.py` records, or shell-expanded seed lists to the c33_64 batch service
rate.

## Interpretation

The mapped capacity slack is positive, but that proves only that the already
measured and mapped production slice lies inside the measured capacity region.
It does not prove that the full production load is stabilizable.

The mapped-capacity slack is now much tighter than before Module59 because the
SimpleSAC slice uses a minimum completed profile-1 lower-service rate.  Module60
adds another conservative completed-history profile-1 lower-service point but
does not further reduce the mapped delta below the SimpleSAC bottleneck:

```text
strict mapped delta = 0.001497772
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
freqduet_cpu_ablation|c_65p
transit_freqhrl_cpu_validation|c_le2
freqduet_cpu_ablation|c_9_16 residual command shapes
sumo_eval_cpu|c_le2 residual command shapes
transit_misc_cpu|c_le2
freqduet_cpu_ablation|c_17_32 residual command shapes
bamor_cpu_training|c_9_16
bamor_cpu_training|c_le2
bamor_cpu_training|c_17_32
freqduet_cpu_ablation|c_3_8 residual native/control command shapes
freqduet_cpu_ablation|c_33_64 residual direct-runner/single-command shapes
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
