# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6488 | 3017 | 3471 | 0.465012 | NA | false |
| `raw_history_all` | `representative` | 6488 | 5110 | 1378 | 0.787608 | NA | false |
| `attempted_production` | `strict` | 4698 | 1987 | 2711 | 0.422946 | NA | false |
| `attempted_production` | `representative` | 4698 | 3937 | 761 | 0.838016 | NA | false |
| `completed_active_production` | `strict` | 3136 | 1696 | 1440 | 0.540816 | NA | false |
| `completed_active_production` | `representative` | 3136 | 2717 | 419 | 0.866390 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3136
mapped_count = 2717
measurement_required_count = 419
mapped_fraction = 0.8663903061224489
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 819 | 0.261161 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:146, bapr_v15:103, CS-BAPR:20, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 327 | 0.104273 | BAMOR:327 |
| `cpu_heavy_local_bench` | `mapped` | 202 | 0.064413 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 172 | 0.054847 | FreqDuet:129, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 152 | 0.048469 | RE-SAC:81, Asumption Agent:52, CFCMT:11, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 122 | 0.038903 | FreqDuet:102, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.033163 | TransitDuet:104 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.028699 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.027742 | freqduet:87 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.026148 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.024554 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 77 | 0.024554 | TransitDuet:56, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.023278 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.020089 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.018495 | BAMOR:58 |
| `scheduler_control_plane` | `measurement_required` | 56 | 0.017857 | scheduleurm:55, sched-hpc-e2e-20260522-214808-175687:1 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 53 | 0.016901 | TransitDuet:49, FreqHRL:4 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.015944 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.012755 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 38 | 0.012117 | TransitDuet:30, offline-sumo:5, python:2, BAMOR:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
