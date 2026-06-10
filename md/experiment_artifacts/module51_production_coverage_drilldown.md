# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6564 | 3118 | 3446 | 0.475015 | NA | false |
| `raw_history_all` | `representative` | 6564 | 5223 | 1341 | 0.795704 | NA | false |
| `attempted_production` | `strict` | 4776 | 2072 | 2704 | 0.433836 | NA | false |
| `attempted_production` | `representative` | 4776 | 4034 | 742 | 0.844640 | NA | false |
| `completed_active_production` | `strict` | 3214 | 1774 | 1440 | 0.551960 | NA | false |
| `completed_active_production` | `representative` | 3214 | 2807 | 407 | 0.873367 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3214
mapped_count = 2807
measurement_required_count = 407
mapped_fraction = 0.873366521468575
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 831 | 0.258556 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:146, bapr_v15:103, CS-BAPR:32, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 367 | 0.114188 | BAMOR:367 |
| `cpu_heavy_local_bench` | `mapped` | 202 | 0.062850 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 172 | 0.053516 | FreqDuet:129, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 158 | 0.049160 | RE-SAC:81, Asumption Agent:58, CFCMT:11, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 122 | 0.037959 | FreqDuet:102, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.032358 | TransitDuet:104 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.028002 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.027069 | freqduet:87 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.025513 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.023958 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 77 | 0.023958 | TransitDuet:56, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.022713 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.019602 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 60 | 0.018668 | TransitDuet:56, FreqHRL:4 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.018046 | BAMOR:58 |
| `scheduler_control_plane` | `measurement_required` | 56 | 0.017424 | scheduleurm:55, sched-hpc-e2e-20260522-214808-175687:1 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.015557 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.012446 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `bamor_diagnostic_shard_c9_16_completed_history` | `mapped` | 34 | 0.010579 | BAMOR:34 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
