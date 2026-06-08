# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5212 | 1269 | 3943 | 0.243477 | NA | false |
| `raw_history_all` | `representative` | 5212 | 2857 | 2355 | 0.548158 | NA | false |
| `attempted_production` | `strict` | 3717 | 411 | 3306 | 0.110573 | NA | false |
| `attempted_production` | `representative` | 3717 | 1869 | 1848 | 0.502825 | NA | false |
| `completed_active_production` | `strict` | 2469 | 393 | 2076 | 0.159174 | NA | false |
| `completed_active_production` | `representative` | 2469 | 1183 | 1286 | 0.479141 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2469
mapped_count = 1183
measurement_required_count = 1286
mapped_fraction = 0.4791413527744026
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 946 | 0.383151 | TransitDuet:319, freqduet:220, BAMOR:183, ZSW_platform:42, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.303767 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 149 | 0.060348 | RE-SAC:82, Asumption Agent:48, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.050628 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.044552 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032402 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031187 | RE-SAC:72, sensing-compressibility-pems:5 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.025516 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.021871 | SimpleSAC:54 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016201 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.016201 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 18 | 0.007290 | scheduleurm:17, sched-hpc-e2e-20260522-214808-175687:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006480 | H2Oplus:14, SimpleSAC:2 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000405 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
