# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 659
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 7.000000 | 64.000000 | 128.000000 |
| `ram_mb` | 5.000000 | 4722.000 | 65536.000 | 256000.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `freqduet_cpu_ablation|c_3_8` | 102 | 0.154780 | 5.000000 | freqduet:85, TransitDuet:12, FreqHRLNative:5 |
| `freqduet_cpu_ablation|c_33_64` | 62 | 0.094082 | 45.500000 | FreqHRL:27, TransitDuet:16, freqduet:15, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.091047 | 95.500000 | TransitDuet:26, FreqHRLNative:21, FreqDuet:11, freq_transitduet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.089530 | 2.000000 | TransitDuet:59 |
| `freqduet_cpu_ablation|c_9_16` | 56 | 0.084977 | 16.000000 | TransitDuet:40, freqduet:12, FreqHRL:4 |
| `sumo_eval_cpu|c_le2` | 51 | 0.077390 | 1.000000 | CFCMT:30, offline-sumo:9, config:5, H2Oplus:2, SimpleSAC:2, Nature_Emissions_gpu1_balanced_p100_20260605_153448:2, ZSW_platform:1 |
| `transit_misc_cpu|c_le2` | 47 | 0.071320 | 1.000000 | TransitDuet:47 |
| `freqduet_cpu_ablation|c_17_32` | 46 | 0.069803 | 30.000000 | freqduet:16, FreqDuet:16, FreqHRL:9, TransitDuet:5 |
| `bamor_cpu_training|c_9_16` | 43 | 0.065250 | 11.000000 | BAMOR:43 |
| `freqduet_cpu_ablation|c_le2` | 39 | 0.059181 | 1.000000 | TransitDuet:23, freqduet:11, python:2, freq_transitduet:1, md:1, FreqHRLNative:1 |
| `bamor_cpu_training|c_le2` | 28 | 0.042489 | 1.000000 | BAMOR:28 |
| `bamor_cpu_training|c_17_32` | 17 | 0.025797 | 23.000000 | BAMOR:17 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.019727 | 4.000000 | TransitDuet:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.018209 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.009105 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.007587 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.007587 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.004552 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.004552 | 3.000000 | TransitDuet:3 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.003035 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_3_8', 'freqduet_cpu_ablation|c_33_64', 'freqduet_cpu_ablation|c_65p', 'transit_freqhrl_cpu_validation|c_le2', 'freqduet_cpu_ablation|c_9_16', 'sumo_eval_cpu|c_le2', 'transit_misc_cpu|c_le2', 'freqduet_cpu_ablation|c_17_32']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
