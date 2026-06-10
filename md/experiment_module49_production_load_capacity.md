# Module49 Production-Load Capacity Attempt

Date: 2026-06-11

This module attempts to move from a declared finite-slice load model to a
production-history load certificate.  It reads Scheduleurm task records from
`~/.claude/scheduler/queue_archive.jsonl` and `~/.claude/scheduler/queue.json`,
estimates arrival load over a 30-day window, maps tasks into measured service
buckets, and solves the capacity-slack LP on the measured action slice.

## Artifacts

```text
algorithm/experiments/production_load_certificate.py
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_strict.md
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module49_production_load_representative.md
```

## Strict Measured-Bucket Result

Strict mapping now includes the Module56 production sub-bucket
`freqduet_cpu_ablation_c17_32` and the Module57 exact-config direct-runner
sub-slice `freqduet_runner_v3_allfreq_alllayers_c9_16`.  Module58 adds
`freqduet_cpu_ablation_c9_16`, with per-record units parsed from production
commands.  It still only counts records backed by a measured service curve or a
ScheduleurmBench measured bucket.
Module59 adds a conservative profile-1 completed-history certificate for
`sumo_eval_simple_sac_c_le2`.
Module60 adds a conservative profile-1 completed-history certificate for
parseable `run_freqduet_ablation.py` records inside `freqduet_cpu_ablation|c_3_8`.
Module61 adds the analogous completed-history certificate for parseable
`run_freqduet_ablation.py` records inside `freqduet_cpu_ablation|c_33_64`.
Module62 adds a completed-history portfolio lower-service certificate for
direct `runner_v3.py` records inside `freqduet_cpu_ablation|c_le2`.
Module63 adds a seed-range completed-history lower-service certificate for
Transit/FreqHRL native promotion validation records inside the c17_32 residual.
Module64 adds a training-step completed-history lower-service certificate for
BAMOR c3_8 CPU training records with parseable work units.
Module65 adds a simulated-second completed-history lower-service certificate
for ZSW TSP/SUMO c_le2 runner records with explicit `--duration`.
Module66 adds a c3_8 direct `runner_v3.py` completed-history certificate.
Module67 refines the broad BAMOR c3_8 class into script-level lower-service
classes so the production load certificate does not apply the slowest
`train_compare_baselines.py` lower bound to faster Mujoco and diagnostic-shard
tasks.
Module68 adds script-parseable c33_64 native-promotion seed-unit classes and
separates the two single-seed smoke/fix records from batch native validation.
Module69 closes the high-CPU c65p residual by splitting it into three strict
completed-history classes: `run_freqduet_ablation.py`, promoted ep100 shell
batches, and native-promotion validation.
Module70 closes the low-CPU Transit/FreqHRL validation/control slice with six
strict completed-history classes: trading sweep, trading policy, transit
surrogate, native promotion, native control, and a singleton import-smoke class.
Module71 closes the previous c9_16 residual by splitting it into three
completed-history classes: the slow bounded_wait_nofinal_v14 native-promotion
stress profile, the residual c9_16 native-promotion profiles, and residual
direct runner_v3 configs.  The bounded/residual split is required because the
single coarse c9_16 native class makes the mapped capacity LP infeasible.
Module72 closes the CFCMT portion of the low-CPU SUMO/evaluation bucket by
splitting 30 completed-active records into six script-level service classes:
feed conversion, environment validation, SUMO generation, snapshot generation,
policy rollout, and traffic-signal phase2.  This split is required because
snapshot and policy-rollout commands contain `sumo_generation` in input paths,
but their progress units and lower-service rates are not SUMO-generation units.
Module74 closes the c17_32 residual command shapes by splitting them into direct
`runner_v3.py`, `run_freqduet_paper_longtrain_matrix.sh`, and residual
native-promotion classes.  The paper-longtrain class uses one completed shard as
the progress unit because those production commands used `--skip-existing`.
Module75 closes the BAMOR c9_16 CPU-training residual by splitting it into
script-level completed-history classes for `train_compare_baselines.py`,
`train_bamor_mujoco.py`, and `run_bamor_diagnostic_shard.py`.
Module76 closes most of the c_le2 residual previously shown under
`freqduet_cpu_ablation|c_le2`: FreqDuet ablation, baseline-rule, preflight/env
checks, Transit/FreqHRL analysis-matrix jobs, and Transit/FreqHRL merge jobs are
separated into five completed-history service classes.  The two auto-adopt spin
helpers remain unmeasured.
Module77 closes the BAMOR c_le2 production-training residual by splitting
`train_compare_baselines.py`, `train_bamor_mujoco.py`, and
`run_bamor_diagnostic_shard.py` into separate profile-1 completed-history
training-step service classes.  Some production commands specify `--device cuda`
even though the scheduler record has zero estimated VRAM, so these rows are kept
as production-history service certificates rather than pure CPU microbenchmarks.
Module78 closes the previous `sumo_eval_cpu|c_le2` residual by splitting
offline-sumo eval scripts, H2Oplus/SimpleSAC complex shell eval jobs, ZSW metrics
parsing, RESCO config/main.py runs, and Nature-emissions extraction/SUMO
singletons.  It uses one completed production command as the service unit for
each class so resume, `--skip_existing`, and shell-loop semantics are not
expanded into artificial work units.
Module79 closes the previous `freqduet_cpu_ablation|c_3_8` first-probe blocker
by identifying it as Transit/FreqHRL native work rather than FreqDuet ablation,
then splitting it into persistent-stress native promotion, native real-demand
batch validation, and alighting-safe/rescue shard validation.
Module80 closes the previous `bamor_cpu_training|c_17_32` first-probe blocker
for the command shapes present in production by splitting it into
`train_bamor_mujoco.py` and `run_bamor_diagnostic_shard.py` completed-history
training-step service classes.  It deliberately does not claim a c17_32
`train_compare_baselines.py` class because that command shape is absent from the
current bucket.
Module81 closes the direct `runner_v3.py` command shape inside the previous
`freqduet_cpu_ablation|c_33_64` residual by adding a separate completed-history
episode service class.  It does not merge these records with c33_64
`run_freqduet_ablation.py` or native-promotion classes.
Module82 closes the previous `sumo_eval_cpu|c_3_8` blocker by splitting it into
four completed-history service classes: CFCMT snapshot generation, CFCMT
SUMO/traffic pytest jobs, CFCMT traffic-signal phase1, and ZSW M21 SUMO
runner records.  The CFCMT snapshot class uses parsed snapshot-window units,
pytest and phase1 use one completed command as the unit, and ZSW M21 uses
parsed simulated seconds from `--duration`.
Module83 closes the next Transit/FreqHRL c3_8 residual by splitting it into six
completed-history service classes: public CSV market evaluation, pressure-matrix
merge, policy-entry train/eval, PPO surrogate train/eval, pytest/unittest jobs,
and native-promotion shard merge.
Module84 closes the next Transit/FreqHRL c17_32 residual by splitting it into
four completed-history service classes: trading pressure-test matrix,
promotion-recovery validation, demand-estimator validation, and gap-closure PPO
surrogate validation.  Pressure, demand, and gap-closure use parsed work units;
promotion-recovery uses one completed command because the scheduler record has
no stable exposed seed/step counter.
Module85 closes the next c9_16 shell-shard native-promotion residual by adding
a wait-credit v39 service class with statically parsed shell arithmetic
seed-index ranges.
Module86 closes the offline-sumo c33_64 eval-command residual with the same
conservative service semantics as Module78: one completed production eval
command is the progress unit because the records use `--skip_existing`/resume
semantics and cannot be safely expanded into item or checkpoint counts.

```text
record_count_window = 6626
mapped_task_count = 3201
mapped_fraction = 0.483097
unmapped_task_count = 3425
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
action_generation = dominating_product_action_certificate
full_action_count = 404352
action_count_evaluated = 1
```

Full estimated-load tables are generated in
`md/experiment_artifacts/module49_production_load_strict.md`.  The Module83-86
strict rows most recently added to the mapped slice are:

| Workload | Count | Lambda |
|---|---:|---:|
| `transit_trading_public_csv_c3_8_completed_history` | 4 | 0.003472222 |
| `transit_trading_pressure_merge_c3_8_completed_history` | 3 | 0.000001157 |
| `transit_trading_policy_c3_8_completed_history` | 1 | 0.011250000 |
| `transit_surrogate_c3_8_completed_history` | 1 | 0.008333333 |
| `transit_freqhrl_tests_c3_8_completed_history` | 8 | 0.000003086 |
| `transit_native_merge_c3_8_completed_history` | 1 | 0.000000386 |
| `transit_trading_pressure_matrix_c17_32_completed_history` | 5 | 0.600000000 |
| `transit_trading_promotion_recovery_c17_32_completed_history` | 1 | 0.000000386 |
| `transit_demand_estimator_c17_32_completed_history` | 10 | 0.024444444 |
| `transit_gap_closure_c17_32_completed_history` | 8 | 0.546666667 |
| `transit_native_promotion_c9_16_wait_credit_shell_completed_history` | 6 | 0.000032407 |
| `offline_sumo_eval_c33_64_completed_history` | 15 | 0.000005787 |

Capacity LP:

```text
delta = 0.000011136
status = optimal
```

For reproducibility, the strict artifact now avoids enumerating the entire
404,352-action product when the service map is the current independent
per-workload profile product.  It constructs the explicit full-product action
that chooses each workload's maximum measured lower-service profile and solves
the same slack check on that single action.  This is a valid sufficient
certificate because that action is a member of the full product action set; it
does not change the measured service map or the theorem assumptions.

## Representative Mapping Result

Representative mapping additionally assigns production RE-SAC/BAPR-like GPU RL
jobs to the measured `hybrid_rl_resac_ant` service bucket and CPU analysis/audit
jobs to the measured local CPU-heavy bucket.  These representative assignments
are diagnostic unless backed by separate equivalence or service-measurement
certificates.

```text
record_count_window = 6626
mapped_task_count = 5308
representative_mapped_task_count = 2107
mapped_fraction = 0.801087
strict_mapped_fraction = 0.483097
unmapped_task_count = 1318
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Full representative-load tables are generated in
`md/experiment_artifacts/module49_production_load_representative.md`.
Module71 through Module80 add the same strict rows shown above; representative
assignments remain diagnostic, not theorem-grade.
Module73 does not alter this raw-window LP.  It tightens the separate Module51
controlled-production population by excluding unobservable external
auto-adopted stdin/wait-for processes from the theorem-facing arrival stream.
The representative raw-window artifact has been regenerated after Module86, but
it remains diagnostic.  Use the strict Module49 artifact above for the current
proof-facing mapped capacity slice.

Capacity LP:

```text
delta = 0.000011136
status = optimal
```

## Interpretation

This closes a narrower but important question: the observed mapped production
load is comfortably inside the currently measured service-action slice, the
Module56 FreqDuet c17_32 sub-bucket is theorem-grade in strict mapping, the
Module57 exact direct-runner config is theorem-grade, Module58 adds a
theorem-grade c9_16 ablation command-shape slice with parsed production units,
Module59 adds a conservative completed-history SimpleSAC c_le2 eval slice, and
Modules60 and 61 add conservative completed-history c3_8 and c33_64 FreqDuet
ablation slices.  Module62 adds the c_le2 direct-runner completed-history
portfolio slice.  Module63 adds the native-promotion seed-range completed-history
slice.  Module64 adds the BAMOR c3_8 CPU-training completed-history slice.
Module65 adds the ZSW TSP/SUMO c_le2 completed-history slice.  Module66 adds
the c3_8 direct FreqDuet runner completed-history slice.  Module67 then splits
the broad BAMOR c3_8 class into script-level service classes; without this
split, the latest raw-window BAMOR load would exceed the broad class's slowest
profile-1 lower-service point.  Module68 closes the c33_64 native-promotion
batch seed-unit class and isolates single-seed smoke/fix measurements into a
separate service class.  Module69 closes c65p ablation, promoted ep100, and
native-promotion residual classes with separated conservative lower-service
points.  The mapped LP is still positive but tight because the completed-history
slices use minimum completed profile-1 lower-service points.
Module70 adds low-CPU Transit/FreqHRL completed-history lower-service points and
removes the former `transit_freqhrl_cpu_validation|c_le2` blocker from the
remaining first-probe order.
Module71 removes the previous `freqduet_cpu_ablation|c_9_16` first-probe
blocker by certifying 7 bounded-wait native-promotion records, 40 residual
native-promotion records, and 12 residual runner_v3 records in the
completed-active view.  A coarse single native c9_16 class is explicitly not
used because its slowest lower-service point does not dominate the aggregate
class load.
Module72 maps the CFCMT part of `sumo_eval_cpu|c_le2`: 5 feed-conversion
records, 5 environment-validation records, 4 SUMO-generation records, 5
snapshot-generation records, 10 policy-rollout records, and 1 traffic-signal
phase2 singleton.  It does not close offline-sumo, H2Oplus, RESCO/config, ZSW
metrics-parser, or Nature-emissions SUMO records.
Module74 maps 16 completed-active c17_32 direct runner records, 16 completed
paper-longtrain shards, and 14 residual native-promotion records.
Module75 maps 6 c9_16 BAMOR train-compare records, 3 c9_16 BAMOR Mujoco
records, and 34 c9_16 BAMOR diagnostic-shard records with script-specific
training-step lower-service rates.  This removes the previous top BAMOR c9_16
CPU-training residual without charging fast diagnostic-shard tasks to the
slower train-compare service class.
Module76 maps the c_le2 residual command shapes that have clear completed
progress units: 7 raw-window FreqDuet ablation records, 5 baseline-rule records,
14 preflight/env-check records, 14 Transit/FreqHRL analysis-matrix records, and
8 Transit/FreqHRL merge records.  It deliberately leaves the two
`freqduet_autoadopt_spin.py` helpers unmeasured because their stable progress
unit is not part of the scheduler-controlled workload model.
Module77 maps 7 raw-window BAMOR c_le2 train-compare records, 16 BAMOR c_le2
Mujoco records, and 3 BAMOR c_le2 diagnostic-shard records with script-specific
training-step lower-service rates.  The split is service-semantic rather than
cosmetic: the three command shapes have materially different completed-history
rates, and the `--device cuda` flag on some records means the certificate must be
read as a production-history row, not a generic CPU-only benchmark.
Module82 maps the c3_8 SUMO block that previously led the remaining Module53
manifest: 1 CFCMT snapshot-generation record, 4 CFCMT pytest records, 1 CFCMT
traffic-signal phase1 record, and 6 ZSW M21 records.  It deliberately excludes
CFCMT local snapshot validation and non-M21 ZSW c3_8 commands until those
command shapes receive their own service certificates.
Module83 maps the next c3_8 Transit/FreqHRL residual: public-market CSV eval,
pressure-matrix merge, policy-entry train/eval, PPO surrogate train/eval,
Transit/FreqHRL test jobs, and native-promotion shard merge.  Failed/cancelled
same-shape attempts can be classified for raw load accounting, but only `done`
records with positive wall-clock duration are used as lower-service samples.
Module84 maps the next c17_32 Transit/FreqHRL residual: pressure-test matrix,
promotion-recovery validation, demand-estimator validation, and gap-closure
validation.  The strict raw-window LP counts all same-shape arrivals in the
30-day raw history, while the lower-service rows are selected only from completed
records with positive wall-clock duration.
Module85 maps the c9_16 wait-credit v39 native-promotion shell shards with a
static parser for literal shell arithmetic assignments.  This removes the
previous `freqduet_cpu_ablation|c_9_16` manifest blocker without charging the
rows to a FreqDuet ablation service class.
Module86 maps c33_64 offline-sumo eval records to a separate completed-history
service class.  It uses one completed eval command as the work unit and does not
expand `--skip_existing` item ranges into artificial service.
Module78 maps 11 raw-window offline-sumo eval records, 5 H2Oplus/SimpleSAC
complex shell eval records, 1 ZSW metrics-parser record, 5 RESCO config/main.py
records, and 2 Nature-emissions records split into extraction and direct SUMO
execution.  The mapped LP remains positive at `0.000011136`,
because the offline-sumo lower-service point is deliberately one completed eval
command over the slowest completed wall-clock duration.
Module79 maps 7 raw-window c3_8 native-promotion persistent-stress records, 28
c3_8 native real-demand batch records, and 14 c3_8 alighting-safe/rescue records
with parsed seed/native-control units.  This removes the former
`freqduet_cpu_ablation|c_3_8` blocker without charging Transit native validation
to a FreqDuet ablation service class.
Module80 maps 14 raw-window c17_32 BAMOR diagnostic-shard records and 3 c17_32
BAMOR Mujoco records using script-specific completed-history training-step
lower-service rates.  This removes the former `bamor_cpu_training|c_17_32`
first-probe blocker while keeping absent c17_32 train-compare work unclaimed.
Module81 maps 32 raw-window c33_64 direct FreqDuet `runner_v3.py` records with
explicit episode units and a conservative completed-history lower-service rate.
This removes the direct-runner part of the previous
`freqduet_cpu_ablation|c_33_64` residual without charging it to the c33_64
ablation or native-promotion service classes.

It does not close the full production theorem claim.  The remaining blockers
are empirical coverage blockers:

```text
strict unmapped: 3425 / 6626 tasks
representative unmapped: 1318 / 6626 tasks
representative-mapped but not theorem-grade: 2107 tasks
```

The next global-closure step remains service coverage: add measured buckets for
the major remaining CPU/SUMO/transit workloads and either prove or measure
bucket equivalence for the representative GPU RL and CPU mappings.
