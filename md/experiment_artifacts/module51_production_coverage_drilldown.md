# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5408 | 1641 | 3767 | 0.303439 | NA | false |
| `raw_history_all` | `representative` | 5408 | 3365 | 2043 | 0.622226 | NA | false |
| `attempted_production` | `strict` | 3912 | 767 | 3145 | 0.196063 | NA | false |
| `attempted_production` | `representative` | 3912 | 2359 | 1553 | 0.603016 | NA | false |
| `completed_active_production` | `strict` | 2553 | 681 | 1872 | 0.266745 | NA | false |
| `completed_active_production` | `representative` | 2553 | 1503 | 1050 | 0.588719 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2553
mapped_count = 1503
measurement_required_count = 1050
mapped_fraction = 0.5887191539365453
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.293772 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 704 | 0.275754 | TransitDuet:256, freqduet:139, BAMOR:88, ZSW_platform:42, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27 |
| `generic_cpu_python` | `measurement_required` | 150 | 0.058754 | RE-SAC:82, Asumption Agent:49, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 136 | 0.053271 | FreqDuet:93, freqduet:36, TransitDuet:7 |
| `bamor_cpu_training_c3_8_completed_history` | `mapped` | 122 | 0.047787 | BAMOR:122 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.043087 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.032902 | freqduet:81, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 81 | 0.031727 | CFCMT:59, Asumption Agent:19, RE-SAC:2, BAPR:1 |
| `artifact_io_control` | `measurement_required` | 77 | 0.030161 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 72 | 0.028202 | BAPR:35, sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, bapr_v15:2, freqduet:1 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 71 | 0.027810 | TransitDuet:71 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.024677 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.021152 | SimpleSAC:54 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.015668 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 22 | 0.008617 | scheduleurm:21, sched-hpc-e2e-20260522-214808-175687:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006267 | H2Oplus:14, SimpleSAC:2 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000392 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
