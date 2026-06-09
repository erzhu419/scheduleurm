# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5980 | 2415 | 3565 | 0.403846 | NA | false |
| `raw_history_all` | `representative` | 5980 | 4311 | 1669 | 0.720903 | NA | false |
| `attempted_production` | `strict` | 4294 | 1457 | 2837 | 0.339311 | NA | false |
| `attempted_production` | `representative` | 4294 | 3222 | 1072 | 0.750349 | NA | false |
| `completed_active_production` | `strict` | 2840 | 1242 | 1598 | 0.437324 | NA | false |
| `completed_active_production` | `representative` | 2840 | 2150 | 690 | 0.757042 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2840
mapped_count = 2150
measurement_required_count = 690
mapped_fraction = 0.7570422535211268
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 787 | 0.277113 | RE-SAC-JMLR:340, RE-SAC:210, BAPR:134, bapr_v15:103 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 320 | 0.112676 | TransitDuet:116, BAMOR:90, freqduet:42, FreqDuet:16, offline-sumo:14, FreqHRL:9, CFCMT:6, ZSW_platform:6 |
| `generic_cpu_python` | `measurement_required` | 153 | 0.053873 | RE-SAC:82, Asumption Agent:51, CFCMT:12, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 153 | 0.053873 | FreqDuet:110, freqduet:36, TransitDuet:7 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 152 | 0.053521 | BAMOR:152 |
| `cpu_heavy_local_bench` | `mapped` | 121 | 0.042606 | BAPR:85, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 116 | 0.040845 | FreqDuet:96, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 91 | 0.032042 | TransitDuet:91 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 86 | 0.030282 | freqduet:86 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 85 | 0.029930 | freqduet:82, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.028873 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.027113 | RE-SAC:72, sensing-compressibility-pems:5 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.022183 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.020423 | BAMOR:58 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 55 | 0.019366 | FreqHRL:27, TransitDuet:24, transit_hrl:2, FreqHRLNative:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.019014 | SimpleSAC:54 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.017606 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 49 | 0.017254 | TransitDuet:28, FreqHRLNative:21 |
| `scheduler_control_plane` | `measurement_required` | 42 | 0.014789 | scheduleurm:41, sched-hpc-e2e-20260522-214808-175687:1 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.014085 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
