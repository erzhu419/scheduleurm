# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 5
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 2.000000 | 7.000000 | 15.000000 | 15.000000 |
| `ram_mb` | 94.000000 | 528.000000 | 625.000000 | 630.000000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.400000 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 2 | 0.400000 | 2.000000 | TransitDuet:2 |
| `bamor_cpu_training|c_3_8` | 1 | 0.200000 | 7.000000 | BAMOR:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['transit_freqhrl_cpu_validation|c_9_16', 'transit_freqhrl_cpu_validation|c_le2', 'bamor_cpu_training|c_3_8']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
