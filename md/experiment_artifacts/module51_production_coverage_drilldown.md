# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6583 | 3144 | 3439 | 0.477594 | NA | false |
| `raw_history_all` | `representative` | 6583 | 5250 | 1333 | 0.797509 | NA | false |
| `attempted_production` | `strict` | 4799 | 2098 | 2701 | 0.437174 | NA | false |
| `attempted_production` | `representative` | 4799 | 4061 | 738 | 0.846218 | NA | false |
| `completed_active_production` | `strict` | 3211 | 1775 | 1436 | 0.552787 | NA | false |
| `completed_active_production` | `representative` | 3211 | 2808 | 403 | 0.874494 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3211
mapped_count = 2808
measurement_required_count = 403
mapped_fraction = 0.8744939271255061
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 831 | 0.258798 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:146, bapr_v15:103, CS-BAPR:32, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 367 | 0.114295 | BAMOR:367 |
| `cpu_heavy_local_bench` | `mapped` | 202 | 0.062909 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 172 | 0.053566 | FreqDuet:129, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 160 | 0.049829 | RE-SAC:81, Asumption Agent:60, CFCMT:11, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 121 | 0.037683 | FreqDuet:101, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.032389 | TransitDuet:104 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.028029 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.027094 | freqduet:87 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.025537 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.023980 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 77 | 0.023980 | TransitDuet:56, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.022734 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.019620 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.018063 | BAMOR:58 |
| `scheduler_control_plane` | `measurement_required` | 56 | 0.017440 | scheduleurm:55, sched-hpc-e2e-20260522-214808-175687:1 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 54 | 0.016817 | TransitDuet:50, FreqHRL:4 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.015571 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.012457 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `bamor_diagnostic_shard_c9_16_completed_history` | `mapped` | 34 | 0.010589 | BAMOR:34 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
