# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5877 | 2038 | 3839 | 0.346776 | NA | false |
| `raw_history_all` | `representative` | 5877 | 3927 | 1950 | 0.668198 | NA | false |
| `attempted_production` | `strict` | 4195 | 1139 | 3056 | 0.271514 | NA | false |
| `attempted_production` | `representative` | 4195 | 2898 | 1297 | 0.690822 | NA | false |
| `completed_active_production` | `strict` | 2755 | 956 | 1799 | 0.347005 | NA | false |
| `completed_active_production` | `representative` | 2755 | 1867 | 888 | 0.677677 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2755
mapped_count = 1867
measurement_required_count = 888
mapped_fraction = 0.6776769509981851
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 790 | 0.286751 | RE-SAC-JMLR:340, RE-SAC:210, BAPR:137, bapr_v15:103 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 532 | 0.193103 | TransitDuet:246, BAMOR:89, freqduet:54, CFCMT:36, FreqHRLNative:29, FreqDuet:27, offline-sumo:14, FreqHRL:13 |
| `generic_cpu_python` | `measurement_required` | 153 | 0.055535 | RE-SAC:82, Asumption Agent:51, CFCMT:12, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 153 | 0.055535 | FreqDuet:110, freqduet:36, TransitDuet:7 |
| `cpu_heavy_local_bench` | `mapped` | 121 | 0.043920 | BAPR:85, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.039927 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 104 | 0.037750 | BAMOR:104 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 86 | 0.031216 | freqduet:86 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.030490 | freqduet:81, freq_transitduet:3 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 83 | 0.030127 | TransitDuet:83 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.029764 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.027949 | RE-SAC:72, sensing-compressibility-pems:5 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.022868 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.021053 | BAMOR:58 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.019601 | SimpleSAC:54 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.018149 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 45 | 0.016334 | FreqHRL:27, TransitDuet:16, transit_hrl:2 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.014519 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 28 | 0.010163 | scheduleurm:27, sched-hpc-e2e-20260522-214808-175687:1 |
| `bamor_diagnostic_shard_c3_8_completed_history` | `mapped` | 25 | 0.009074 | BAMOR:25 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
