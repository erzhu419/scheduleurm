# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6853 | 3715 | 3138 | 0.542098 | NA | false |
| `raw_history_all` | `representative` | 6853 | 5938 | 915 | 0.866482 | NA | false |
| `attempted_production` | `strict` | 5088 | 2674 | 2414 | 0.525550 | NA | false |
| `attempted_production` | `representative` | 5088 | 4751 | 337 | 0.933766 | NA | false |
| `completed_active_production` | `strict` | 3378 | 2349 | 1029 | 0.695382 | NA | false |
| `completed_active_production` | `representative` | 3378 | 3378 | 0 | 1.000000 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3378
mapped_count = 3378
measurement_required_count = 0
mapped_fraction = 1.0
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 840 | 0.248668 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:143, bapr_v15:103, CS-BAPR:44, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 488 | 0.144464 | BAMOR:488 |
| `cpu_heavy_local_bench` | `mapped` | 189 | 0.055950 | BAPR:164, sensing-compressibility-pems:13, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 174 | 0.051510 | FreqDuet:131, freqduet:36, TransitDuet:7 |
| `resac_review5_jax_train_fabric_completed_history` | `mapped` | 151 | 0.044701 | RE-SAC:151 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 121 | 0.035820 | FreqDuet:101, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.030787 | TransitDuet:104 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 92 | 0.027235 | TransitDuet:66, FreqHRLNative:21, freq_hrl:5 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.026643 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.025755 | freqduet:87 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 76 | 0.022499 | TransitDuet:44, FreqHRL:27, transit_hrl:2, FreqHRLNative:2, freq_hrl:1 |
| `cfcmt_cpu_eval_completed_history` | `mapped` | 71 | 0.021018 | CFCMT:71 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.018650 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 58 | 0.017170 | TransitDuet:54, FreqHRL:4 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.017170 | BAMOR:58 |
| `scheduleurm_control_plane_completed_history` | `mapped` | 55 | 0.016282 | scheduleurm:55 |
| `assumption_agent_unittest_completed_history` | `mapped` | 51 | 0.015098 | Asumption Agent:51 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.014802 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.011841 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `bamor_diagnostic_shard_c9_16_completed_history` | `mapped` | 34 | 0.010065 | BAMOR:34 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
