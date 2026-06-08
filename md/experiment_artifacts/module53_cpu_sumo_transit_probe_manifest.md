# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 1009
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 8.000000 | 48.000000 | 128.000000 |
| `ram_mb` | 5.000000 | 7769.000 | 98304.000 | 256000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `freqduet_cpu_ablation|c_33_64` | 125 | 0.123885 | 48.000000 | FreqDuet:54, FreqHRL:27, freqduet:22, TransitDuet:18, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation|c_le2` | 123 | 0.121903 | 1.000000 | freqduet:92, TransitDuet:23, freq_transitduet:4, python:2, md:1, FreqHRLNative:1 |
| `freqduet_cpu_ablation|c_17_32` | 116 | 0.114965 | 32.000000 | TransitDuet:75, freqduet:16, FreqDuet:16, FreqHRL:9 |
| `sumo_eval_cpu|c_le2` | 100 | 0.099108 | 1.000000 | ZSW_platform:37, CFCMT:30, zsw_tsp_m0_gpu1:14, offline-sumo:9, config:5, H2Oplus:2, SimpleSAC:2, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation|c_3_8` | 96 | 0.095144 | 5.000000 | freqduet:85, TransitDuet:6, FreqHRLNative:5 |
| `bamor_cpu_training|c_3_8` | 95 | 0.094153 | 8.000000 | BAMOR:95 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.059465 | 95.500000 | TransitDuet:26, FreqHRLNative:21, FreqDuet:11, freq_transitduet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.058474 | 2.000000 | TransitDuet:59 |
| `freqduet_cpu_ablation|c_9_16` | 50 | 0.049554 | 16.000000 | TransitDuet:34, freqduet:12, FreqHRL:4 |
| `transit_misc_cpu|c_le2` | 47 | 0.046581 | 1.000000 | TransitDuet:47 |
| `bamor_cpu_training|c_9_16` | 43 | 0.042616 | 11.000000 | BAMOR:43 |
| `bamor_cpu_training|c_le2` | 28 | 0.027750 | 1.000000 | BAMOR:28 |
| `bamor_cpu_training|c_17_32` | 17 | 0.016848 | 23.000000 | BAMOR:17 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.012884 | 4.000000 | TransitDuet:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.011893 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_17_32` | 7 | 0.006938 | 32.000000 | TransitDuet:7 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.004955 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.004955 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.002973 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.002973 | 3.000000 | TransitDuet:3 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.001982 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_33_64', 'freqduet_cpu_ablation|c_le2', 'freqduet_cpu_ablation|c_17_32', 'sumo_eval_cpu|c_le2', 'freqduet_cpu_ablation|c_3_8', 'bamor_cpu_training|c_3_8', 'freqduet_cpu_ablation|c_65p', 'transit_freqhrl_cpu_validation|c_le2']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
