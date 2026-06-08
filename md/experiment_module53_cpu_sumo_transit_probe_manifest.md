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

## Status After Module56

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
completed_active_production mapped count = 125
raw_history_all mapped count = 156
```

The current Module53 manifest has therefore been regenerated over the remaining
unmeasured `cpu_sumo_transit_eval_or_control` records.

## Current Remaining Bucket

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 1208
cpu_cores median = 8
cpu_cores p90 = 48
cpu_cores max = 128
ram_mb median = 7110
ram_mb p90 = 65536
ram_mb max = 256000
theorem_status = measurement_required
```

## Top Remaining Sub-Buckets

| Sub-Bucket | Count | Fraction |
|---|---:|---:|
| `freqduet_cpu_ablation|c_9_16` | 161 | 0.133278 |
| `sumo_eval_cpu|c_le2` | 154 | 0.127483 |
| `freqduet_cpu_ablation|c_3_8` | 136 | 0.112583 |
| `freqduet_cpu_ablation|c_33_64` | 125 | 0.103477 |
| `freqduet_cpu_ablation|c_le2` | 123 | 0.101821 |
| `freqduet_cpu_ablation|c_17_32` | 116 | 0.096026 |
| `bamor_cpu_training|c_3_8` | 95 | 0.078642 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.049669 |

## Next Probe Order

The regenerated manifest recommends this first pass over the remaining bucket:

```text
freqduet_cpu_ablation|c_9_16
sumo_eval_cpu|c_le2
freqduet_cpu_ablation|c_3_8
freqduet_cpu_ablation|c_33_64
freqduet_cpu_ablation|c_le2
freqduet_cpu_ablation|c_17_32
bamor_cpu_training|c_3_8
freqduet_cpu_ablation|c_65p
```

Each remaining sub-bucket still needs progress-bearing service curves over:

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = local_cpu, direct_hpc_cpu_node
```

## Interpretation

Module53 no longer says the first production CPU/SUMO measurement is merely a
plan.  The exact `run_freqduet_ablation.py` c17_32 slice is measured and
strictly mapped.  The global theorem remains open because 1208 completed/active
production records in the CPU/SUMO/transit family still require measured curves
or equivalence certificates, including 116 residual `freqduet_cpu_ablation|c_17_32`
records with different command shapes.
