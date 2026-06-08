# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5154 | 963 | 4191 | 0.186845 | 0.319917853 | false |
| `raw_history_all` | `representative` | 5154 | 2551 | 2603 | 0.494955 | 0.272849952 | false |
| `attempted_production` | `strict` | 3690 | 131 | 3559 | 0.035501 | 0.334609211 | false |
| `attempted_production` | `representative` | 3690 | 1589 | 2101 | 0.430623 | 0.291245013 | false |
| `completed_active_production` | `strict` | 2442 | 125 | 2317 | 0.051188 | 0.334609211 | false |
| `completed_active_production` | `representative` | 2442 | 915 | 1527 | 0.374693 | 0.311461063 | false |

## Completed/Active Production Obligations

```text
record_count = 2442
mapped_count = 915
measurement_required_count = 1527
mapped_fraction = 0.3746928746928747
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1208 | 0.494676 | TransitDuet:342, freqduet:265, BAMOR:177, FreqDuet:172, SimpleSAC:56, ZSW_platform:42, FreqHRL:40, CFCMT:36 |
| `hybrid_rl_resac_ant` | `mapped` | 750 | 0.307125 | RE-SAC-JMLR:340, RE-SAC:210, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 144 | 0.058968 | RE-SAC:82, Asumption Agent:43, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, Nature_Emissions_gpu1_balanced_p100_20260605_153448:1 |
| `freqduet_cpu_ablation_c17_32` | `mapped` | 125 | 0.051188 | FreqDuet:82, freqduet:36, TransitDuet:7 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.032760 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.031532 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 40 | 0.016380 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:4, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006552 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 2 | 0.000819 | sched-hpc-e2e-20260522-214808-175687:1, scheduleurm:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
