# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5176 | 964 | 4212 | 0.186244 | 0.319917853 | false |
| `raw_history_all` | `representative` | 5176 | 2552 | 2624 | 0.493045 | 0.272849952 | false |
| `attempted_production` | `strict` | 3697 | 132 | 3565 | 0.035705 | 0.334609211 | false |
| `attempted_production` | `representative` | 3697 | 1590 | 2107 | 0.430078 | 0.291245013 | false |
| `completed_active_production` | `strict` | 2449 | 126 | 2323 | 0.051450 | 0.334609211 | false |
| `completed_active_production` | `representative` | 2449 | 916 | 1533 | 0.374030 | 0.311461063 | false |

## Completed/Active Production Obligations

```text
record_count = 2449
mapped_count = 916
measurement_required_count = 1533
mapped_fraction = 0.3740302164148632
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1211 | 0.494488 | TransitDuet:342, freqduet:264, BAMOR:181, FreqDuet:172, SimpleSAC:56, ZSW_platform:42, FreqHRL:40, CFCMT:36 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.306247 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 146 | 0.059616 | RE-SAC:82, Asumption Agent:45, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.051041 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032666 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031441 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016333 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006533 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 3 | 0.001225 | scheduleurm:2, sched-hpc-e2e-20260522-214808-175687:1 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000408 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
