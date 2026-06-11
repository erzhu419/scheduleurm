# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6957 | 4825 | 2132 | 0.693546 | NA | false |
| `raw_history_all` | `representative` | 6957 | 6034 | 923 | 0.867328 | NA | false |
| `attempted_production` | `strict` | 5172 | 3764 | 1408 | 0.727765 | NA | false |
| `attempted_production` | `representative` | 5172 | 4827 | 345 | 0.933295 | NA | false |
| `completed_active_production` | `strict` | 3437 | 3437 | 0 | 1.000000 | NA | true |
| `completed_active_production` | `representative` | 3437 | 3437 | 0 | 1.000000 | NA | true |

## Completed/Active Production Obligations

```text
record_count = 3437
mapped_count = 3437
measurement_required_count = 0
mapped_fraction = 1.0
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 499 | 0.145185 | BAMOR:499 |
| `hybrid_rl_resac_jmlr_project_fabric_completed_history` | `mapped` | 340 | 0.098923 | RE-SAC-JMLR:340 |
| `hybrid_rl_resac_project_fabric_completed_history` | `mapped` | 208 | 0.060518 | RE-SAC:208 |
| `cpu_heavy_local_fabric_completed_history` | `mapped` | 189 | 0.054990 | BAPR:164, sensing-compressibility-pems:13, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 181 | 0.052662 | FreqDuet:138, freqduet:36, TransitDuet:7 |
| `resac_review5_jax_train_fabric_completed_history` | `mapped` | 151 | 0.043934 | RE-SAC:151 |
| `hybrid_rl_bapr_project_fabric_completed_history` | `mapped` | 143 | 0.041606 | BAPR:143 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 122 | 0.035496 | FreqDuet:102, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 107 | 0.031132 | TransitDuet:104, freq_hrl:3 |
| `hybrid_rl_bapr_v15_project_fabric_completed_history` | `mapped` | 103 | 0.029968 | bapr_v15:103 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.026186 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.025313 | freqduet:87 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 87 | 0.025313 | TransitDuet:66, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 75 | 0.021821 | TransitDuet:44, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `cfcmt_cpu_eval_completed_history` | `mapped` | 71 | 0.020658 | CFCMT:71 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 71 | 0.020658 | TransitDuet:54, freq_hrl:13, FreqHRL:4 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.018330 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `hybrid_rl_cs_bapr_project_fabric_completed_history` | `mapped` | 59 | 0.017166 | CS-BAPR:59 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.016875 | BAMOR:58 |
| `scheduleurm_control_plane_completed_history` | `mapped` | 58 | 0.016875 | scheduleurm:58 |

## Interpretation

The reviewer-facing completed-active production population has full strict measured-bucket coverage. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
