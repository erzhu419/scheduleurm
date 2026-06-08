# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5229 | 1481 | 3748 | 0.283228 | NA | false |
| `raw_history_all` | `representative` | 5229 | 3071 | 2158 | 0.587302 | NA | false |
| `attempted_production` | `strict` | 3735 | 609 | 3126 | 0.163052 | NA | false |
| `attempted_production` | `representative` | 3735 | 2069 | 1666 | 0.553949 | NA | false |
| `completed_active_production` | `strict` | 2487 | 548 | 1939 | 0.220346 | NA | false |
| `completed_active_production` | `representative` | 2487 | 1340 | 1147 | 0.538802 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2487
mapped_count = 1340
measurement_required_count = 1147
mapped_fraction = 0.5388017691998391
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 803 | 0.322879 | TransitDuet:254, BAMOR:189, freqduet:139, ZSW_platform:42, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.301568 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 149 | 0.059912 | RE-SAC:82, Asumption Agent:48, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.050261 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.044230 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.033776 | freqduet:81, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032167 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.030961 | RE-SAC:72, sensing-compressibility-pems:5 |
| `transit_native_promotion_c17_32_seedrange_completed_history` | `mapped` | 71 | 0.028548 | TransitDuet:71 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.025332 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.021713 | SimpleSAC:54 |
| `cpu_heavy_local_bench` | `mapped` | 42 | 0.016888 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, BAPR:5, sensing-compressibility-v10k:4, bapr_v15:2, freqduet:1 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.016084 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 22 | 0.008846 | scheduleurm:21, sched-hpc-e2e-20260522-214808-175687:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006433 | H2Oplus:14, SimpleSAC:2 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000402 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
