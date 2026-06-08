# Module53 Production Bucket Probe Manifest

```text
bucket = cpu_sumo_transit_eval_or_control
record_count = 1245
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
| `freqduet_cpu_ablation|c_17_32` | 212 | 0.170281 | 32.000000 | FreqDuet:81, TransitDuet:70, freqduet:52, FreqHRL:9 |
| `freqduet_cpu_ablation|c_9_16` | 153 | 0.122892 | 14.000000 | FreqDuet:90, TransitDuet:45, freqduet:14, FreqHRL:4 |
| `sumo_eval_cpu|c_le2` | 153 | 0.122892 | 2.000000 | SimpleSAC:56, ZSW_platform:37, CFCMT:30, zsw_tsp_m0_gpu1:14, offline-sumo:9, config:5, H2Oplus:2 |
| `freqduet_cpu_ablation|c_3_8` | 134 | 0.107631 | 5.000000 | freqduet:121, TransitDuet:6, FreqHRLNative:5, freq_transitduet:1, FreqDuet:1 |
| `freqduet_cpu_ablation|c_33_64` | 124 | 0.099598 | 48.000000 | FreqDuet:54, FreqHRL:27, freqduet:22, TransitDuet:17, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation|c_le2` | 122 | 0.097992 | 1.000000 | freqduet:91, TransitDuet:23, freq_transitduet:4, python:2, md:1, FreqHRLNative:1 |
| `bamor_cpu_training|c_3_8` | 80 | 0.064257 | 8.000000 | BAMOR:80 |
| `freqduet_cpu_ablation|c_65p` | 60 | 0.048193 | 95.500000 | TransitDuet:26, FreqHRLNative:21, FreqDuet:11, freq_transitduet:2 |
| `transit_freqhrl_cpu_validation|c_le2` | 59 | 0.047390 | 2.000000 | TransitDuet:59 |
| `transit_misc_cpu|c_le2` | 47 | 0.037751 | 1.000000 | TransitDuet:47 |
| `bamor_cpu_training|c_9_16` | 29 | 0.023293 | 13.000000 | BAMOR:29 |
| `bamor_cpu_training|c_17_32` | 13 | 0.010442 | 21.000000 | BAMOR:13 |
| `sumo_eval_cpu|c_3_8` | 12 | 0.009639 | 3.500000 | CFCMT:6, ZSW_platform:5, zsw_tsp_m0_gpu1:1 |
| `transit_freqhrl_cpu_validation|c_3_8` | 12 | 0.009639 | 4.000000 | TransitDuet:12 |
| `bamor_cpu_training|c_le2` | 10 | 0.008032 | 1.000000 | BAMOR:10 |
| `transit_freqhrl_cpu_validation|c_17_32` | 7 | 0.005622 | 32.000000 | TransitDuet:7 |
| `sumo_eval_cpu|c_33_64` | 5 | 0.004016 | 37.000000 | offline-sumo:5 |
| `transit_freqhrl_cpu_validation|c_9_16` | 5 | 0.004016 | 9.000000 | TransitDuet:5 |
| `transit_misc_cpu|c_17_32` | 3 | 0.002410 | 23.000000 | TransitDuet:3 |
| `transit_misc_cpu|c_3_8` | 3 | 0.002410 | 3.000000 | TransitDuet:3 |
| `transit_freqhrl_cpu_validation|c_33_64` | 2 | 0.001606 | 54.500000 | TransitDuet:2 |

## Probe Grid

```text
task_concurrency_profiles = [1, 2, 4, 8]
node_targets = ['local_cpu', 'direct_hpc_cpu_node']
first_probe_order = ['freqduet_cpu_ablation|c_17_32', 'sumo_eval_cpu|c_le2', 'freqduet_cpu_ablation|c_9_16', 'freqduet_cpu_ablation|c_3_8', 'freqduet_cpu_ablation|c_33_64', 'freqduet_cpu_ablation|c_le2', 'bamor_cpu_training|c_3_8', 'freqduet_cpu_ablation|c_65p']
```

## Interpretation

This manifest extracts real production templates for a missing coverage bucket. It does not certify service until the listed sub-buckets receive progress-bearing service curves.
