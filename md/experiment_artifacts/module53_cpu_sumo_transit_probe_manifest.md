# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 471
theorem_status = measurement_required
```

## Resource Summary

| Resource | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| `cpu_cores` | 1.000000 | 3.000000 | 30.000000 | 61.000000 |
| `ram_mb` | 5.000000 | 2541.000 | 65536.000 | 131072.000 |

## Sub-Buckets

| Sub-Bucket | Count | Fraction | CPU Median | Top Projects |
|---|---:|---:|---:|---|
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.125265 | 2.000000 | TransitDuet:59 |
| `freqduet_cpu_ablation|c_9_16` | 56 | 0.118896 | 16.000000 | TransitDuet:40, freqduet:12, FreqHRL:4 |
| `sumo_eval_cpu|c_le2` | 51 | 0.108280 | 1.000000 | CFCMT:30, offline-sumo:9, config:5, H2Oplus:2, SimpleSAC:2, Nature_Emissions_gpu1_balanced_p100_20260605_153448:2, ZSW_platform:1 |
| `transit_misc_cpu|c_le2` | 48 | 0.101911 | 1.000000 | TransitDuet:48 |
| `freqduet_cpu_ablation|c_17_32` | 46 | 0.097665 | 30.000000 | freqduet:16, FreqDuet:16, FreqHRL:9, TransitDuet:5 |
| `bamor_cpu_training|c_9_16` | 43 | 0.091295 | 11.000000 | BAMOR:43 |
| `freqduet_cpu_ablation|c_le2` | 39 | 0.082803 | 1.000000 | TransitDuet:23, freqduet:11, python:2, freq_transitduet:1, md:1, FreqHRLNative:1 |
| `bamor_cpu_training|c_le2` | 28 | 0.059448 | 1.000000 | BAMOR:28 |
| `freqduet_cpu_ablation|c_3_8` | 18 | 0.038217 | 8.000000 | TransitDuet:13, FreqHRLNative:5 |
| `bamor_cpu_training|c_17_32` | 17 | 0.036093 | 23.000000 | BAMOR:17 |
| `freqduet_cpu_ablation|c_33_64` | 15 | 0.031847 | 48.000000 | freqduet:15 |
| `transit_freqhrl_cpu_validation|c_3_8` | 13 | 0.027601 | 4.000000 | TransitDuet:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.025478 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_17_32` | 6 | 0.012739 | 32.000000 | TransitDuet:6 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.010616 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.010616 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.006369 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.006369 | 3.000000 | TransitDuet:3 |
| `bamor_cpu_training|c_3_8` | 2 | 0.004246 | 4.500000 | BAMOR:2 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.004246 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['transit_freqhrl_cpu_validation|c_le2', 'freqduet_cpu_ablation|c_9_16', 'sumo_eval_cpu|c_le2', 'transit_misc_cpu|c_le2', 'freqduet_cpu_ablation|c_17_32', 'bamor_cpu_training|c_9_16', 'freqduet_cpu_ablation|c_le2', 'bamor_cpu_training|c_le2']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
