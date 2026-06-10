# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6656 | 3228 | 3428 | 0.484976 | NA | false |
| `raw_history_all` | `representative` | 6656 | 5335 | 1321 | 0.801532 | NA | false |
| `attempted_production` | `strict` | 4873 | 2174 | 2699 | 0.446132 | NA | false |
| `attempted_production` | `representative` | 4873 | 4138 | 735 | 0.849169 | NA | false |
| `completed_active_production` | `strict` | 3280 | 1849 | 1431 | 0.563720 | NA | false |
| `completed_active_production` | `representative` | 3280 | 2882 | 398 | 0.878659 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3280
mapped_count = 2882
measurement_required_count = 398
mapped_fraction = 0.8786585365853659
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 831 | 0.253354 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:146, bapr_v15:103, CS-BAPR:32, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 430 | 0.131098 | BAMOR:430 |
| `cpu_heavy_local_bench` | `mapped` | 202 | 0.061585 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 178 | 0.054268 | FreqDuet:135, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 162 | 0.049390 | RE-SAC:81, Asumption Agent:62, CFCMT:11, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 121 | 0.036890 | FreqDuet:101, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.031707 | TransitDuet:104 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.027439 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.026524 | freqduet:87 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.025000 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.023476 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 77 | 0.023476 | TransitDuet:56, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.022256 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.019207 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.017683 | BAMOR:58 |
| `scheduler_control_plane` | `measurement_required` | 56 | 0.017073 | scheduleurm:55, sched-hpc-e2e-20260522-214808-175687:1 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 54 | 0.016463 | TransitDuet:50, FreqHRL:4 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.015244 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.012195 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `bamor_diagnostic_shard_c9_16_completed_history` | `mapped` | 34 | 0.010366 | BAMOR:34 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
