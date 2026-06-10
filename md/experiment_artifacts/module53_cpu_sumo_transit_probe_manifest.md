# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 10
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 4.500000 | 48.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 528.000000 | 64000.000 | 64000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `freqduet_cpu_ablation|c_le2` | 3 | 0.300000 | 1.000000 | python:2, TransitDuet:1 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.200000 | 54.500000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.200000 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 2 | 0.200000 | 2.000000 | TransitDuet:2 |
| `bamor_cpu_training|c_3_8` | 1 | 0.100000 | 7.000000 | BAMOR:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_le2', 'transit_freqhrl_cpu_validation|c_33_64', 'transit_freqhrl_cpu_validation|c_9_16', 'transit_freqhrl_cpu_validation|c_le2', 'bamor_cpu_training|c_3_8']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
