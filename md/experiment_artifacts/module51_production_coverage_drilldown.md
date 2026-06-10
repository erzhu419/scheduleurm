# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6179 | 2774 | 3405 | 0.448940 | NA | false |
| `raw_history_all` | `representative` | 6179 | 4699 | 1480 | 0.760479 | NA | false |
| `attempted_production` | `strict` | 4400 | 1771 | 2629 | 0.402500 | NA | false |
| `attempted_production` | `representative` | 4400 | 3565 | 835 | 0.810227 | NA | false |
| `completed_active_production` | `strict` | 2910 | 1510 | 1400 | 0.518900 | NA | false |
| `completed_active_production` | `representative` | 2910 | 2447 | 463 | 0.840893 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2910
mapped_count = 2447
measurement_required_count = 463
mapped_fraction = 0.840893470790378
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 790 | 0.271478 | RE-SAC-JMLR:340, RE-SAC:210, BAPR:137, bapr_v15:103 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 215 | 0.073883 | BAMOR:215 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 161 | 0.055326 | FreqDuet:118, freqduet:36, TransitDuet:7 |
| `cpu_heavy_local_bench` | `mapped` | 147 | 0.050515 | BAPR:111, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2 |
| `generic_cpu_python` | `measurement_required` | 144 | 0.049485 | RE-SAC:81, Asumption Agent:43, CFCMT:11, sensing-compressibility-v10k:7, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 116 | 0.039863 | FreqDuet:96, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 97 | 0.033333 | TransitDuet:97 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 94 | 0.032302 | TransitDuet:38, BAMOR:17, freqduet:15, CFCMT:6, offline-sumo:5, ZSW_platform:5, FreqHRLNative:5, python:2 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 86 | 0.029553 | freqduet:86 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 86 | 0.029553 | freqduet:83, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.028179 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.026460 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.025086 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.021649 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.019931 | BAMOR:58 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.018557 | SimpleSAC:54 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.017182 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `scheduler_control_plane` | `measurement_required` | 50 | 0.017182 | scheduleurm:49, sched-hpc-e2e-20260522-214808-175687:1 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 49 | 0.016838 | TransitDuet:28, FreqHRLNative:21 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 46 | 0.015808 | TransitDuet:42, FreqHRL:4 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
