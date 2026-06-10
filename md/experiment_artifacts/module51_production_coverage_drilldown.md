# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6428 | 2961 | 3467 | 0.460641 | NA | false |
| `raw_history_all` | `representative` | 6428 | 5017 | 1411 | 0.780492 | NA | false |
| `attempted_production` | `strict` | 4654 | 1956 | 2698 | 0.420284 | NA | false |
| `attempted_production` | `representative` | 4654 | 3886 | 768 | 0.834981 | NA | false |
| `completed_active_production` | `strict` | 3101 | 1671 | 1430 | 0.538858 | NA | false |
| `completed_active_production` | `representative` | 3101 | 2675 | 426 | 0.862625 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3101
mapped_count = 2675
measurement_required_count = 426
mapped_fraction = 0.8626249596904224
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 803 | 0.258949 | RE-SAC-JMLR:340, RE-SAC:210, BAPR:137, bapr_v15:103, CS-BAPR:13 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 327 | 0.105450 | BAMOR:327 |
| `cpu_heavy_local_bench` | `mapped` | 201 | 0.064818 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 169 | 0.054499 | FreqDuet:126, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 144 | 0.046437 | RE-SAC:81, Asumption Agent:43, CFCMT:11, sensing-compressibility-v10k:7, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 116 | 0.037407 | FreqDuet:96, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 103 | 0.033215 | TransitDuet:103 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 86 | 0.027733 | freqduet:86 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 86 | 0.027733 | freqduet:83, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.026443 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.024831 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.023541 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.020316 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.018704 | BAMOR:58 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 56 | 0.018059 | TransitDuet:22, freqduet:15, CFCMT:6, offline-sumo:5, ZSW_platform:5, python:2, zsw_tsp_m0_gpu1:1 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.017414 | SimpleSAC:54 |
| `scheduler_control_plane` | `measurement_required` | 51 | 0.016446 | scheduleurm:50, sched-hpc-e2e-20260522-214808-175687:1 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.016124 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 49 | 0.015801 | TransitDuet:28, FreqHRLNative:21 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 47 | 0.015156 | TransitDuet:43, FreqHRL:4 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
