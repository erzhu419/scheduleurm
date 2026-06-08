# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5168 | 807 | 4361 | 0.156153 | 0.319917853 | false |
| `raw_history_all` | `representative` | 5168 | 2585 | 2583 | 0.500193 | 0.266954890 | false |
| `attempted_production` | `strict` | 3735 | 0 | 3735 | 0.000000 | 0.334609211 | false |
| `attempted_production` | `representative` | 3735 | 1648 | 2087 | 0.441232 | 0.285349952 | false |
| `completed_active_production` | `strict` | 2495 | 0 | 2495 | 0.000000 | 0.334609211 | false |
| `completed_active_production` | `representative` | 2495 | 948 | 1547 | 0.379960 | 0.306553655 | false |

## Completed/Active Production Obligations

```text
record_count = 2495
mapped_count = 948
measurement_required_count = 1547
mapped_fraction = 0.37995991983967936
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1227 | 0.491784 | TransitDuet:323, freqduet:300, FreqDuet:237, BAMOR:116, SimpleSAC:56, ZSW_platform:42, FreqHRL:40, CFCMT:36 |
| `hybrid_rl_resac_ant` | `mapped` | 909 | 0.364329 | RE-SAC:369, RE-SAC-JMLR:340, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 136 | 0.054509 | RE-SAC:82, Asumption Agent:35, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1, BAMOR:1 |
| `cpu_eval_generic` | `measurement_required` | 89 | 0.035671 | CFCMT:59, Asumption Agent:19, BAMOR:9, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.030862 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 39 | 0.015631 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:3, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006413 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 2 | 0.000802 | sched-hpc-e2e-20260522-214808-175687:1, scheduleurm:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
