# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 1211
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 8.000000 | 48.000000 | 128.000000 |
| `ram_mb` | 5.000000 | 8192.000 | 65536.000 | 256000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `freqduet_cpu_ablation|c_9_16` | 160 | 0.132122 | 14.000000 | FreqDuet:90, TransitDuet:53, freqduet:13, FreqHRL:4 |
| `sumo_eval_cpu|c_le2` | 154 | 0.127168 | 2.000000 | SimpleSAC:56, ZSW_platform:37, CFCMT:30, zsw_tsp_m0_gpu1:14, offline-sumo:9, config:5, H2Oplus:2, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation|c_3_8` | 136 | 0.112304 | 5.000000 | freqduet:121, TransitDuet:8, FreqHRLNative:5, freq_transitduet:1, FreqDuet:1 |
| `freqduet_cpu_ablation|c_33_64` | 125 | 0.103220 | 48.000000 | FreqDuet:54, FreqHRL:27, freqduet:22, TransitDuet:18, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation|c_le2` | 123 | 0.101569 | 1.000000 | freqduet:92, TransitDuet:23, freq_transitduet:4, python:2, md:1, FreqHRLNative:1 |
| `freqduet_cpu_ablation|c_17_32` | 116 | 0.095789 | 32.000000 | TransitDuet:75, freqduet:16, FreqDuet:16, FreqHRL:9 |
| `bamor_cpu_training|c_3_8` | 95 | 0.078448 | 8.000000 | BAMOR:95 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.049546 | 95.500000 | TransitDuet:26, FreqHRLNative:21, FreqDuet:11, freq_transitduet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.048720 | 2.000000 | TransitDuet:59 |
| `transit_misc_cpu|c_le2` | 47 | 0.038811 | 1.000000 | TransitDuet:47 |
| `bamor_cpu_training|c_9_16` | 43 | 0.035508 | 11.000000 | BAMOR:43 |
| `bamor_cpu_training|c_le2` | 26 | 0.021470 | 1.000000 | BAMOR:26 |
| `bamor_cpu_training|c_17_32` | 17 | 0.014038 | 23.000000 | BAMOR:17 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.010735 | 4.000000 | TransitDuet:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.009909 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_17_32` | 7 | 0.005780 | 32.000000 | TransitDuet:7 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.004129 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.004129 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.002477 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.002477 | 3.000000 | TransitDuet:3 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.001652 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_9_16', 'sumo_eval_cpu|c_le2', 'freqduet_cpu_ablation|c_3_8', 'freqduet_cpu_ablation|c_33_64', 'freqduet_cpu_ablation|c_le2', 'freqduet_cpu_ablation|c_17_32', 'bamor_cpu_training|c_3_8', 'freqduet_cpu_ablation|c_65p']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
