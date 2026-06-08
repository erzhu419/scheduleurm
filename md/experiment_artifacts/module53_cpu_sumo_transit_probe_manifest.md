# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 862
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
| `freqduet_cpu_ablation|c_17_32` | 116 | 0.134571 | 32.000000 | TransitDuet:75, freqduet:16, FreqDuet:16, FreqHRL:9 |
| `sumo_eval_cpu|c_le2` | 100 | 0.116009 | 1.000000 | ZSW_platform:37, CFCMT:30, zsw_tsp_m0_gpu1:14, offline-sumo:9, config:5, H2Oplus:2, SimpleSAC:2, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation|c_3_8` | 96 | 0.111369 | 5.000000 | freqduet:85, TransitDuet:6, FreqHRLNative:5 |
| `bamor_cpu_training|c_3_8` | 95 | 0.110209 | 8.000000 | BAMOR:95 |
| `freqduet_cpu_ablation|c_33_64` | 62 | 0.071926 | 45.500000 | FreqHRL:27, TransitDuet:16, freqduet:15, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.069606 | 95.500000 | TransitDuet:26, FreqHRLNative:21, FreqDuet:11, freq_transitduet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.068445 | 2.000000 | TransitDuet:59 |
| `freqduet_cpu_ablation|c_9_16` | 50 | 0.058005 | 16.000000 | TransitDuet:34, freqduet:12, FreqHRL:4 |
| `transit_misc_cpu|c_le2` | 47 | 0.054524 | 1.000000 | TransitDuet:47 |
| `bamor_cpu_training|c_9_16` | 43 | 0.049884 | 11.000000 | BAMOR:43 |
| `freqduet_cpu_ablation|c_le2` | 39 | 0.045244 | 1.000000 | TransitDuet:23, freqduet:11, python:2, freq_transitduet:1, md:1, FreqHRLNative:1 |
| `bamor_cpu_training|c_le2` | 28 | 0.032483 | 1.000000 | BAMOR:28 |
| `bamor_cpu_training|c_17_32` | 17 | 0.019722 | 23.000000 | BAMOR:17 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.015081 | 4.000000 | TransitDuet:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.013921 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_17_32` | 7 | 0.008121 | 32.000000 | TransitDuet:7 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.005800 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.005800 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.003480 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.003480 | 3.000000 | TransitDuet:3 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.002320 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_17_32', 'sumo_eval_cpu|c_le2', 'freqduet_cpu_ablation|c_3_8', 'bamor_cpu_training|c_3_8', 'freqduet_cpu_ablation|c_33_64', 'freqduet_cpu_ablation|c_65p', 'transit_freqhrl_cpu_validation|c_le2', 'freqduet_cpu_ablation|c_9_16']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
