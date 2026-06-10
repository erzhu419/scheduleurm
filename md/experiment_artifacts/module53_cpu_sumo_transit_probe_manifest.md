# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 73
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 21.000000 | 48.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 9677.000 | 65536.000 | 65536.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `bamor_cpu_training|c_17_32` | 17 | 0.232877 | 23.000000 | BAMOR:17 |
| `freqduet_cpu_ablation|c_33_64` | 15 | 0.205479 | 48.000000 | freqduet:15 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.164384 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_3_8` | 11 | 0.150685 | 4.000000 | TransitDuet:11 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.082192 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.068493 | 37.000000 | offline-sumo:5 |
| `freqduet_cpu_ablation|c_le2` | 2 | 0.027397 | 1.000000 | python:2 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.027397 | 54.500000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.027397 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 1 | 0.013699 | 2.000000 | TransitDuet:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['bamor_cpu_training|c_17_32', 'freqduet_cpu_ablation|c_33_64', 'sumo_eval_cpu|c_3_8', 'transit_freqhrl_cpu_validation|c_3_8', 'transit_freqhrl_cpu_validation|c_17_32', 'sumo_eval_cpu|c_33_64', 'transit_freqhrl_cpu_validation|c_33_64', 'transit_freqhrl_cpu_validation|c_9_16']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
