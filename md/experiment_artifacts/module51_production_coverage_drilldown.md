# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 6551 | 3082 | 3469 | 0.470463 | NA | false |
| `raw_history_all` | `representative` | 6551 | 5187 | 1364 | 0.791788 | NA | false |
| `attempted_production` | `strict` | 4761 | 2050 | 2711 | 0.430582 | NA | false |
| `attempted_production` | `representative` | 4761 | 4012 | 749 | 0.842680 | NA | false |
| `completed_active_production` | `strict` | 3199 | 1756 | 1443 | 0.548922 | NA | false |
| `completed_active_production` | `representative` | 3199 | 2789 | 410 | 0.871835 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 3199
mapped_count = 2789
measurement_required_count = 410
mapped_fraction = 0.8718349484213817
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 831 | 0.259769 | RE-SAC-JMLR:340, RE-SAC:208, BAPR:146, bapr_v15:103, CS-BAPR:32, sensing-compressibility-v10k:2 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 367 | 0.114723 | BAMOR:367 |
| `cpu_heavy_local_bench` | `mapped` | 202 | 0.063145 | BAPR:165, sensing-compressibility-pems:13, CFCMT:12, sensing-compressibility-v10k:5, TransitDuet:5, bapr_v15:2 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 172 | 0.053767 | FreqDuet:129, freqduet:36, TransitDuet:7 |
| `generic_cpu_python` | `measurement_required` | 155 | 0.048453 | RE-SAC:81, Asumption Agent:55, CFCMT:11, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 122 | 0.038137 | FreqDuet:102, TransitDuet:19, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 104 | 0.032510 | TransitDuet:104 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 90 | 0.028134 | freqduet:87, freq_transitduet:3 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 87 | 0.027196 | freqduet:87 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.025633 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.024070 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c65p_completed_history` | `mapped` | 77 | 0.024070 | TransitDuet:56, FreqHRLNative:21 |
| `transit_native_promotion_c33_64_batch_completed_history` | `mapped` | 73 | 0.022820 | TransitDuet:42, FreqHRL:27, transit_hrl:2, FreqHRLNative:2 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.019694 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.018131 | BAMOR:58 |
| `scheduler_control_plane` | `measurement_required` | 56 | 0.017505 | scheduleurm:55, sched-hpc-e2e-20260522-214808-175687:1 |
| `transit_native_promotion_c9_16_residual_completed_history` | `mapped` | 54 | 0.016880 | TransitDuet:50, FreqHRL:4 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.015630 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.012504 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `bamor_diagnostic_shard_c9_16_completed_history` | `mapped` | 34 | 0.010628 | BAMOR:34 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
