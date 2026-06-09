# Module49 Production-Load Capacity Attempt

Date: 2026-06-09

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

```text
record_count_window = 5935
mapped_task_count = 2353
mapped_fraction = 0.396462
unmapped_task_count = 3582
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Full estimated-load tables are generated in
`md/experiment_artifacts/module49_production_load_strict.md`.  The Module71
strict rows added to the mapped slice are:

| Workload | Count | Lambda |
|---|---:|---:|
| `transit_native_promotion_c9_16_bounded_wait_completed_history` | 9 | 0.000900463 |
| `transit_native_promotion_c9_16_residual_completed_history` | 45 | 0.003028935 |
| `freqduet_runner_v3_c9_16_residual_completed_history` | 13 | 0.000185185 |

Capacity LP:

```text
delta = 0.001497772
status = optimal
```

## Representative Mapping Result

Representative mapping additionally assigns production RE-SAC/BAPR-like GPU RL
jobs to the measured `hybrid_rl_resac_ant` service bucket and CPU analysis/audit
jobs to the measured local CPU-heavy bucket.  These representative assignments
are diagnostic unless backed by separate equivalence or service-measurement
certificates.

```text
record_count_window = 5935
mapped_task_count = 4242
representative_mapped_task_count = 1889
mapped_fraction = 0.714743
strict_mapped_fraction = 0.396462
unmapped_task_count = 1693
mapped_capacity_usable_for_theorem = true
global_coverage_usable_for_theorem = false
usable_for_global_theorem = false
```

Full representative-load tables are generated in
`md/experiment_artifacts/module49_production_load_representative.md`.  Module71
adds the same three strict c9_16 residual rows shown above; representative
assignments remain diagnostic, not theorem-grade.

Capacity LP:

```text
delta = 0.001497772
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

It does not close the full production theorem claim.  The remaining blockers
are empirical coverage blockers:

```text
strict unmapped: 3582 / 5935 tasks
representative unmapped: 1693 / 5935 tasks
representative-mapped but not theorem-grade: 1889 tasks
```

The next global-closure step remains service coverage: add measured buckets for
the major remaining CPU/SUMO/transit workloads and either prove or measure
bucket equivalence for the representative GPU RL and CPU mappings.
