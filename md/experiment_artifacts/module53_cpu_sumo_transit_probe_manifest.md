# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 38
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 14.000000 | 37.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 1073.500 | 32768.000 | 64000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `transit_freqhrl_cpu_validation|c_3_8` | 12 | 0.315789 | 4.000000 | TransitDuet:12 |
| `freqduet_cpu_ablation|c_9_16` | 6 | 0.157895 | 14.000000 | TransitDuet:6 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.157895 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.131579 | 37.000000 | offline-sumo:5 |
| `freqduet_cpu_ablation|c_le2` | 2 | 0.052632 | 1.000000 | python:2 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.052632 | 54.500000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.052632 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 2 | 0.052632 | 2.000000 | TransitDuet:2 |
| `bamor_cpu_training|c_3_8` | 1 | 0.026316 | 7.000000 | BAMOR:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['transit_freqhrl_cpu_validation|c_3_8', 'transit_freqhrl_cpu_validation|c_17_32', 'freqduet_cpu_ablation|c_9_16', 'sumo_eval_cpu|c_33_64', 'transit_freqhrl_cpu_validation|c_33_64', 'transit_freqhrl_cpu_validation|c_9_16', 'freqduet_cpu_ablation|c_le2', 'transit_freqhrl_cpu_validation|c_le2']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
