# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5617 | 1838 | 3779 | 0.327221 | NA | false |
| `raw_history_all` | `representative` | 5617 | 3689 | 1928 | 0.656756 | NA | false |
| `attempted_production` | `strict` | 4119 | 964 | 3155 | 0.234037 | NA | false |
| `attempted_production` | `representative` | 4119 | 2683 | 1436 | 0.651372 | NA | false |
| `completed_active_production` | `strict` | 2679 | 878 | 1801 | 0.327734 | NA | false |
| `completed_active_production` | `representative` | 2679 | 1749 | 930 | 0.652856 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2679
mapped_count = 1749
measurement_required_count = 930
mapped_fraction = 0.6528555431131019
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.279955 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 576 | 0.215006 | TransitDuet:261, BAMOR:89, freqduet:54, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27, offline-sumo:14 |
| `generic_cpu_python` | `measurement_required` | 153 | 0.057111 | RE-SAC:82, Asumption Agent:51, CFCMT:12, sensing-compressibility-v10k:6, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 147 | 0.054871 | FreqDuet:104, freqduet:36, TransitDuet:7 |
| `cpu_heavy_local_bench` | `mapped` | 121 | 0.045166 | BAPR:85, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.041060 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `bamor_mujoco_c3_8_completed_history` | `mapped` | 89 | 0.033221 | BAMOR:89 |
| `freqduet_runner_v3_c3_8_completed_history` | `mapped` | 86 | 0.032102 | freqduet:86 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.031355 | freqduet:81, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 82 | 0.030608 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.028742 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 71 | 0.026502 | TransitDuet:71 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.023516 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `bamor_train_compare_c3_8_completed_history` | `mapped` | 58 | 0.021650 | BAMOR:58 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.020157 | SimpleSAC:54 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.018664 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.014931 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 26 | 0.009705 | scheduleurm:25, sched-hpc-e2e-20260522-214808-175687:1 |
| `bamor_diagnostic_shard_c3_8_completed_history` | `mapped` | 25 | 0.009332 | BAMOR:25 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.005972 | H2Oplus:14, SimpleSAC:2 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
