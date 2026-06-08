# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5201 | 1164 | 4037 | 0.223803 | 0.001497772 | false |
| `raw_history_all` | `representative` | 5201 | 2752 | 2449 | 0.529129 | 0.001497772 | false |
| `attempted_production` | `strict` | 3706 | 308 | 3398 | 0.083108 | 0.001504331 | false |
| `attempted_production` | `representative` | 3706 | 1766 | 1940 | 0.476525 | 0.001504331 | false |
| `completed_active_production` | `strict` | 2458 | 290 | 2168 | 0.117982 | 0.001504331 | false |
| `completed_active_production` | `representative` | 2458 | 1080 | 1378 | 0.439382 | 0.001504331 | false |

## Completed/Active Production Obligations

```text
record_count = 2458
mapped_count = 1080
measurement_required_count = 1378
mapped_fraction = 0.43938161106590723
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1049 | 0.426770 | TransitDuet:323, freqduet:263, BAMOR:183, FreqDuet:82, ZSW_platform:42, FreqHRL:40, CFCMT:36, FreqHRLNative:29 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.305126 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 149 | 0.060618 | RE-SAC:82, Asumption Agent:48, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.050854 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.044752 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032547 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031326 | RE-SAC:72, sensing-compressibility-pems:5 |
| `sumo_eval_simple_sac_c_le2` | `mapped` | 54 | 0.021969 | SimpleSAC:54 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016273 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006509 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 7 | 0.002848 | scheduleurm:6, sched-hpc-e2e-20260522-214808-175687:1 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000407 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
