# Module53 CPU/SUMO/Transit Probe Manifest

Date: 2026-06-08

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

## Status After Module64

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
completed_active_production mapped count = 136
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
completed_active_production mapped count = 71
raw_history_all mapped count = 128
feasible profiles = 1
unit = seed_episode
```

Module64 measured a completed-history profile-1 lower-service point for BAMOR
c3_8 CPU training records with parseable training-step units:

```text
workload_key = bamor_cpu_training_c3_8_completed_history
completed_active_production mapped count = 122
raw_history_all mapped count = 143
feasible profiles = 1
unit = training_step
```

The current Module53 manifest has therefore been regenerated over the remaining
unmeasured `cpu_sumo_transit_eval_or_control` records.

## Current Remaining Bucket

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 704
cpu_cores median = 5
cpu_cores p90 = 60
cpu_cores max = 128
ram_mb median = 4096
ram_mb p90 = 65536
ram_mb max = 256000
theorem_status = measurement_required
```

## Top Remaining Sub-Buckets

| Sub-Bucket | Count | Fraction |
|---|---:|---:|
| `sumo_eval_cpu|c_le2` | 100 | 0.142045 |
| `freqduet_cpu_ablation|c_3_8` | 98 | 0.139205 |
| `freqduet_cpu_ablation|c_33_64` | 62 | 0.088068 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.085227 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.083807 |
| `freqduet_cpu_ablation|c_9_16` | 56 | 0.079545 |
| `transit_misc_cpu|c_le2` | 47 | 0.066761 |
| `freqduet_cpu_ablation|c_17_32` | 46 | 0.065341 |
| `bamor_cpu_training|c_9_16` | 43 | 0.061080 |

## Next Probe Order

The regenerated manifest recommends this first pass over the remaining bucket:

```text
sumo_eval_cpu|c_le2
freqduet_cpu_ablation|c_3_8 residual command shapes
freqduet_cpu_ablation|c_33_64 residual command shapes
freqduet_cpu_ablation|c_65p
transit_freqhrl_cpu_validation|c_le2
freqduet_cpu_ablation|c_9_16 residual command shapes
transit_misc_cpu|c_le2
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
lower-service point.  Module63 maps 71 completed-active Transit
native-promotion c17_32 seed-range records with a profile-1 completed-history
lower-service point.  Module64 maps 122 completed-active BAMOR c3_8 CPU-training
records with a profile-1 completed-history lower-service point.  The global
theorem remains open because 704
completed/active production records in the CPU/SUMO/transit family still
require measured curves or equivalence certificates, including 100
`sumo_eval_cpu|c_le2` records, 98 residual
`freqduet_cpu_ablation|c_3_8` records, 62 residual
`freqduet_cpu_ablation|c_33_64` records, 60
`freqduet_cpu_ablation|c_65p` records, 59
`transit_freqhrl_cpu_validation|c_le2` records, 56 residual
`freqduet_cpu_ablation|c_9_16` records, and 46 residual
`freqduet_cpu_ablation|c_17_32` records with command shapes not covered by
Module56 or Module63.
