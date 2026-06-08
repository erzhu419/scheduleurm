# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5196 | 1093 | 4103 | 0.210354 | 0.319917853 | false |
| `raw_history_all` | `representative` | 5196 | 2681 | 2515 | 0.515974 | 0.272849952 | false |
| `attempted_production` | `strict` | 3701 | 254 | 3447 | 0.068630 | 0.334609211 | false |
| `attempted_production` | `representative` | 3701 | 1712 | 1989 | 0.462578 | 0.291245013 | false |
| `completed_active_production` | `strict` | 2453 | 236 | 2217 | 0.096209 | 0.334609211 | false |
| `completed_active_production` | `representative` | 2453 | 1026 | 1427 | 0.418263 | 0.311461063 | false |

## Completed/Active Production Obligations

```text
record_count = 2453
mapped_count = 1026
measurement_required_count = 1427
mapped_fraction = 0.418263350998777
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1103 | 0.449653 | TransitDuet:323, freqduet:263, BAMOR:183, FreqDuet:82, SimpleSAC:56, ZSW_platform:42, FreqHRL:40, CFCMT:36 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.305748 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 147 | 0.059927 | RE-SAC:82, Asumption Agent:46, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.050958 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `freqduet_cpu_ablation_c9_16` | `mapped` | 110 | 0.044843 | FreqDuet:90, TransitDuet:19, freqduet:1 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032613 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031390 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016307 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006523 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 4 | 0.001631 | scheduleurm:3, sched-hpc-e2e-20260522-214808-175687:1 |
| `freqduet_runner_v3_allfreq_alllayers_c9_16` | `mapped` | 1 | 0.000408 | freqduet:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
