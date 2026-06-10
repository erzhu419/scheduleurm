# Module49 Production-Load Capacity Attempt

Date: 2026-06-10

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

```text
record_count_window = 6486
mapped_task_count = 3016
mapped_fraction = 0.465002
unmapped_task_count = 3470
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Full estimated-load tables are generated in
`md/experiment_artifacts/module49_production_load_strict.md`.  The Module82
strict rows added to the mapped slice are:

| Workload | Count | Lambda |
|---|---:|---:|
| `cfcmt_snapshot_generation_c3_8_completed_history` | 1 | 0.000555556 |
| `cfcmt_pytest_sumo_c3_8_completed_history` | 4 | 0.000001543 |
| `cfcmt_traffic_signal_phase1_c3_8_completed_history` | 1 | 0.000000386 |
| `zsw_m21_sumo_eval_c3_8_completed_history` | 6 | 0.041666667 |

Capacity LP:

```text
delta = 0.000011136
status = optimal
```

## Representative Mapping Result

Representative mapping additionally assigns production RE-SAC/BAPR-like GPU RL
jobs to the measured `hybrid_rl_resac_ant` service bucket and CPU analysis/audit
jobs to the measured local CPU-heavy bucket.  These representative assignments
are diagnostic unless backed by separate equivalence or service-measurement
certificates.

```text
record_count_window = 6400
mapped_task_count = 4990
representative_mapped_task_count = 2064
mapped_fraction = 0.779688
strict_mapped_fraction = 0.457188
unmapped_task_count = 1410
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
The representative raw-window artifact has not been regenerated after
Module82 because the representative LP is diagnostic and can be much larger
than the strict theorem-grade LP.  Use the strict Module49 artifact above for
the current proof-facing mapped capacity slice.

Capacity LP:

```text
delta = 0.000010750
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
Module78 maps 11 raw-window offline-sumo eval records, 5 H2Oplus/SimpleSAC
complex shell eval records, 1 ZSW metrics-parser record, 5 RESCO config/main.py
records, and 2 Nature-emissions records split into extraction and direct SUMO
execution.  The mapped LP remains positive but tightens to `0.000008435`,
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
strict unmapped: 3472 / 6440 tasks
representative unmapped: 1410 / 6400 tasks
representative-mapped but not theorem-grade: 2064 tasks
```

The next global-closure step remains service coverage: add measured buckets for
the major remaining CPU/SUMO/transit workloads and either prove or measure
bucket equivalence for the representative GPU RL and CPU mappings.
