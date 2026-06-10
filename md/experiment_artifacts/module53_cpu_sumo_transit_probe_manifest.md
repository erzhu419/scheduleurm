# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 56
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 16.000000 | 48.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 1517.000 | 19729.000 | 64000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `freqduet_cpu_ablation|c_33_64` | 15 | 0.267857 | 48.000000 | freqduet:15 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.214286 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_3_8` | 11 | 0.196429 | 4.000000 | TransitDuet:11 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.107143 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.089286 | 37.000000 | offline-sumo:5 |
| `freqduet_cpu_ablation|c_le2` | 2 | 0.035714 | 1.000000 | python:2 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.035714 | 54.500000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_9_16` | 2 | 0.035714 | 15.000000 | TransitDuet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 1 | 0.017857 | 2.000000 | TransitDuet:1 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_33_64', 'sumo_eval_cpu|c_3_8', 'transit_freqhrl_cpu_validation|c_3_8', 'transit_freqhrl_cpu_validation|c_17_32', 'sumo_eval_cpu|c_33_64', 'transit_freqhrl_cpu_validation|c_33_64', 'transit_freqhrl_cpu_validation|c_9_16', 'freqduet_cpu_ablation|c_le2']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
