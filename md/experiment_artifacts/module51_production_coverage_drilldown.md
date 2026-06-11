# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6933 | 4803 | 2130 | 0.692774 | NA | false |
| `raw_history_all` | `representative` | 6933 | 6012 | 921 | 0.867157 | NA | false |
| `attempted_production` | `strict` | 5151 | 3746 | 1405 | 0.727237 | NA | false |
| `attempted_production` | `representative` | 5151 | 4809 | 342 | 0.933605 | NA | false |
| `completed_active_production` | `strict` | 3419 | 3419 | 0 | 1.000000 | NA | true |
| `completed_active_production` | `representative` | 3419 | 3419 | 0 | 1.000000 | NA | true |

## Completed/Active Production Obligations

```text
record_count = 3419
mapped_count = 3419
measurement_required_count = 0
mapped_fraction = 1.0
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 488 | 0.142732 | BAMOR:488 |
| `hybrid_rl_resac_jmlr_project_fabric_completed_history` | `mapped` | 340 | 0.099444 | RE-SAC-JMLR:340 |
| `hybrid_rl_resac_project_fabric_completed_history` | `mapped` | 208 | 0.060837 | RE-SAC:208 |
| `cpu_heavy_local_fabric_completed_history` | `mapped` | 189 | 0.055279 | BAPR:164, sensing-compressibility-pems:13, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 178 | 0.052062 | FreqDuet:135, freqduet:36, TransitDuet:7 |
| `resac_review5_jax_train_fabric_completed_history` | `mapped` | 151 | 0.044165 | RE-SAC:151 |
| `hybrid_rl_bapr_project_fabric_completed_history` | `mapped` | 143 | 0.041825 | BAPR:143 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 122 | 0.035683 | FreqDuet:102, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 107 | 0.031296 | TransitDuet:104, freq_hrl:3 |
| `hybrid_rl_bapr_v15_project_fabric_completed_history` | `mapped` | 103 | 0.030126 | bapr_v15:103 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.026323 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.025446 | freqduet:87 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 87 | 0.025446 | TransitDuet:66, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 75 | 0.021936 | TransitDuet:44, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `cfcmt_cpu_eval_completed_history` | `mapped` | 71 | 0.020766 | CFCMT:71 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 71 | 0.020766 | TransitDuet:54, freq_hrl:13, FreqHRL:4 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.018426 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `hybrid_rl_cs_bapr_project_fabric_completed_history` | `mapped` | 59 | 0.017257 | CS-BAPR:59 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.016964 | BAMOR:58 |
| `scheduleurm_control_plane_completed_history` | `mapped` | 57 | 0.016672 | scheduleurm:57 |

## Interpretation

The reviewer-facing completed-active production population has full strict measured-bucket coverage. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
