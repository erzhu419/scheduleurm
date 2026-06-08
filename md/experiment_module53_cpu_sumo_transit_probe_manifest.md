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

## Current Bucket

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 1245
cpu_cores median = 8
cpu_cores p90 = 48
cpu_cores max = 128
ram_mb median = 8192
ram_mb p90 = 65536
ram_mb max = 256000
theorem_status = measurement_required
```

## Top Sub-Buckets

| Sub-Bucket | Count | Fraction |
|---|---:|---:|
| `freqduet_cpu_ablation|c_17_32` | 212 | 0.170281 |
| `freqduet_cpu_ablation|c_9_16` | 153 | 0.122892 |
| `sumo_eval_cpu|c_le2` | 153 | 0.122892 |
| `freqduet_cpu_ablation|c_3_8` | 134 | 0.107631 |
| `freqduet_cpu_ablation|c_33_64` | 124 | 0.099598 |
| `freqduet_cpu_ablation|c_le2` | 122 | 0.097992 |
| `bamor_cpu_training|c_3_8` | 80 | 0.064257 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.048193 |

## First Probe Order

The manifest recommends this first pass:

```text
freqduet_cpu_ablation|c_17_32
sumo_eval_cpu|c_le2
freqduet_cpu_ablation|c_9_16
freqduet_cpu_ablation|c_3_8
freqduet_cpu_ablation|c_33_64
freqduet_cpu_ablation|c_le2
bamor_cpu_training|c_3_8
freqduet_cpu_ablation|c_65p
```

Each sub-bucket still needs progress-bearing service curves over:

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = local_cpu, direct_hpc_cpu_node
```

## Interpretation

This closes the ambiguity in the production coverage blocker.  It does not yet
close the theorem condition, because no service curves have been measured for
these sub-buckets.  The next empirical closure module should implement a
runner/summary format for these CPU/SUMO/transit probes and then add the
measured profiles into the service cache and production capacity certificate.
