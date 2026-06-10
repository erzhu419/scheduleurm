# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 43
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 4.000000 | 37.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 625.000000 | 16000.000 | 64000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `sumo_eval_cpu|c_3_8` | 12 | 0.279070 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_3_8` | 11 | 0.255814 | 4.000000 | TransitDuet:11 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.139535 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.116279 | 37.000000 | offline-sumo:5 |
| `freqduet_cpu_ablation|c_le2` | 2 | 0.046512 | 1.000000 | python:2 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.046512 | 54.500000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.046512 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 2 | 0.046512 | 2.000000 | TransitDuet:2 |
| `bamor_cpu_training|c_3_8` | 1 | 0.023256 | 7.000000 | BAMOR:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['sumo_eval_cpu|c_3_8', 'transit_freqhrl_cpu_validation|c_3_8', 'transit_freqhrl_cpu_validation|c_17_32', 'sumo_eval_cpu|c_33_64', 'transit_freqhrl_cpu_validation|c_33_64', 'transit_freqhrl_cpu_validation|c_9_16', 'freqduet_cpu_ablation|c_le2', 'transit_freqhrl_cpu_validation|c_le2']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
