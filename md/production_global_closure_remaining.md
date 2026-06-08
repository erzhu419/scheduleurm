# Production Global Closure Remaining

Date: 2026-06-08

This note records the current gap after Module59.  It should be read together
with:

```text
md/experiment_module53_cpu_sumo_transit_probe_manifest.md
md/experiment_module56_freqduet_cpu_production_curve.md
md/experiment_module57_freqduet_runner_v3_c9_16_curve.md
md/experiment_module58_freqduet_ablation_c9_16_curve.md
md/experiment_module59_simple_sac_sumo_eval_cle2_completed_history.md
md/or_submission_gap_closure.md
```

## Current State

The global production theorem is not fully closed.

Already closed:

```text
exact slice = run_freqduet_ablation.py within freqduet_cpu_ablation|c_17_32
workload_key = freqduet_cpu_ablation_c17_32
completed-active strict mapped count = 125
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
```

Not yet closed:

```text
completed_active_production records = 2458
strict completed-active mapped count = 290
representative completed-active mapped count = 1080
measurement_required = 1378
cpu_sumo_transit_eval_or_control remaining = 1049 / 2458
```

The original `freqduet_cpu_ablation|c_17_32` group is not fully closed.  Module56
only certifies records that actually invoke `run_freqduet_ablation.py`.  The
remaining `freqduet_cpu_ablation|c_17_32` records include 116 completed/active
tasks with different command shapes, such as native validation or direct
`runner_v3.py` invocations, so the Module56 curve must not be used to certify
them.

The current `freqduet_cpu_ablation|c_9_16` group is still not fully closed.
Module57 certifies only the direct `runner_v3.py` exact config
`configs_freqduet/F_allfreq_alllayers_hiro.yaml`, which accounts for one
completed/active production record.  Module58 certifies 110 c9_16
`run_freqduet_ablation.py` records with parseable units.  The remaining c9_16
records have different command shapes and stay in the Module53 manifest.

## Interpretation

The mapped capacity slack is positive, but that proves only that the already
measured and mapped production slice lies inside the measured capacity region.
It does not prove that the full production load is stabilizable.

The mapped-capacity slack is now much tighter than before Module59 because the
SimpleSAC slice uses a minimum completed profile-1 lower-service rate:

```text
strict mapped delta = 0.001497772
completed-active representative delta = 0.001504331
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
freqduet_cpu_ablation|c_3_8
freqduet_cpu_ablation|c_33_64
freqduet_cpu_ablation|c_le2
freqduet_cpu_ablation|c_17_32 residual command shapes
sumo_eval_cpu|c_le2 residual command shapes
bamor_cpu_training|c_3_8
freqduet_cpu_ablation|c_65p
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
