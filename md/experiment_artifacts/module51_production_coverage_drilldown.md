# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5213 | 1353 | 3860 | 0.259543 | NA | false |
| `raw_history_all` | `representative` | 5213 | 2941 | 2272 | 0.564167 | NA | false |
| `attempted_production` | `strict` | 3719 | 495 | 3224 | 0.133100 | NA | false |
| `attempted_production` | `representative` | 3719 | 1953 | 1766 | 0.525141 | NA | false |
| `completed_active_production` | `strict` | 2471 | 477 | 1994 | 0.193039 | NA | false |
| `completed_active_production` | `representative` | 2471 | 1267 | 1204 | 0.512748 | NA | false |

## Completed/Active Production Obligations

```text
record_count = 2471
mapped_count = 1267
measurement_required_count = 1204
mapped_fraction = 0.5127478753541076
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 862 | 0.348847 | TransitDuet:319, BAMOR:183, freqduet:139, ZSW_platform:42, FreqHRL:40, CFCMT:36, FreqHRLNative:29, FreqDuet:27 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.303521 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 149 | 0.060299 | RE-SAC:82, Asumption Agent:48, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.050587 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.044516 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `freqduet_runner_v3_c_le2_completed_history` | `mapped` | 84 | 0.033994 | freqduet:81, freq_transitduet:3 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032376 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031161 | RE-SAC:72, sensing-compressibility-pems:5 |
| `freqduet_cpu_ablation_c33_64_completed_history` | `mapped` | 63 | 0.025496 | FreqDuet:54, freqduet:7, TransitDuet:2 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.021854 | SimpleSAC:54 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016188 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `freqduet_cpu_ablation_c3_8_completed_history` | `mapped` | 40 | 0.016188 | freqduet:36, TransitDuet:2, freq_transitduet:1, FreqDuet:1 |
| `scheduler_control_plane` | `measurement_required` | 20 | 0.008094 | scheduleurm:19, sched-hpc-e2e-20260522-214808-175687:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006475 | H2Oplus:14, SimpleSAC:2 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000405 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate. This coverage drilldown intentionally does not solve the capacity LP; use Module49 for mapped-slice capacity delta.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
