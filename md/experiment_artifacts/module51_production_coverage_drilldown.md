# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5527 | 1711 | 3816 | 0.309571 | NA | false |
| `raw_history_all` | `representative` | 5527 | 3523 | 2004 | 0.637416 | NA | false |
| `attempted_production` | `strict` | 4029 | 837 | 3192 | 0.207744 | NA | false |
| `attempted_production` | `representative` | 4029 | 2517 | 1512 | 0.624721 | NA | false |
| `completed_active_production` | `strict` | 2589 | 751 | 1838 | 0.290073 | NA | false |
| `completed_active_production` | `representative` | 2589 | 1583 | 1006 | 0.611433 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2589
mapped_count = 1583
measurement_required_count = 1006
mapped_fraction = 0.6114329857087679
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.289687 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 659 | 0.254538 | TransitDuet:260, freqduet:139, BAMOR:88, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27, offline-sumo:14 |
| `generic_cpu_python` | `measurement_required` | 151 | 0.058324 | RE-SAC:82, Asumption Agent:50, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `bamor_cpu_training_c3_8_completed_history` | `mapped` | 142 | 0.054847 | BAMOR:142 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 136 | 0.052530 | FreqDuet:93, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.042487 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.032445 | freqduet:81, freq_transitduet:3 |
| `cpu_heavy_local_bench` | `mapped` | 82 | 0.031672 | BAPR:45, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2, freqduet:1 |
| `cpu_eval_generic` | `measurement_required` | 81 | 0.031286 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:1 |
| `artifact_io_control` | `measurement_required` | 77 | 0.029741 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 71 | 0.027424 | TransitDuet:71 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.024334 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.020857 | SimpleSAC:54 |
| `zsw_tsp_sumo_eval_c_le2_completed_history` | `mapped` | 50 | 0.019312 | ZSW_platform:36, zsw_tsp_m0_gpu1:14 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.015450 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 22 | 0.008497 | scheduleurm:21, sched-hpc-e2e-20260522-214808-175687:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006180 | H2Oplus:14, SimpleSAC:2 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000386 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
