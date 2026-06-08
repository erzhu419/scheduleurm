# Module51 Production Coverage Drilldown

## Population Views

| View | Mapping | Records | Mapped | Unmapped | Mapped Fraction | Delta | Global Theorem |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw_history_all` | `strict` | 5177 | 807 | 4370 | 0.155882 | 0.319917853 | false |
| `raw_history_all` | `representative` | 5177 | 2585 | 2592 | 0.499324 | 0.266954890 | false |
| `attempted_production` | `strict` | 3744 | 0 | 3744 | 0.000000 | 0.334609211 | false |
| `attempted_production` | `representative` | 3744 | 1648 | 2096 | 0.440171 | 0.285349952 | false |
| `completed_active_production` | `strict` | 2504 | 0 | 2504 | 0.000000 | 0.334609211 | false |
| `completed_active_production` | `representative` | 2504 | 948 | 1556 | 0.378594 | 0.306553655 | false |

## Completed/Active Production Obligations

```text
record_count = 2504
mapped_count = 948
measurement_required_count = 1556
mapped_fraction = 0.37859424920127793
```

| Bucket | Status | Count | Fraction | Top Projects |
|---|---|---:|---:|---|
| `cpu_sumo_transit_eval_or_control` | `measurement_required` | 1245 | 0.497204 | TransitDuet:325, freqduet:300, FreqDuet:237, BAMOR:132, SimpleSAC:56, ZSW_platform:42, FreqHRL:40, CFCMT:36 |
| `hybrid_rl_resac_ant` | `mapped` | 909 | 0.363019 | RE-SAC:369, RE-SAC-JMLR:340, bapr_v15:103, BAPR:97 |
| `generic_cpu_python` | `measurement_required` | 136 | 0.054313 | RE-SAC:82, Asumption Agent:36, CFCMT:12, sensing-compressibility-v10k:5, sensing-compressibility-v2k:1 |
| `cpu_eval_generic` | `measurement_required` | 80 | 0.031949 | CFCMT:59, Asumption Agent:19, RE-SAC:2 |
| `artifact_io_control` | `measurement_required` | 77 | 0.030751 | RE-SAC:72, sensing-compressibility-pems:5 |
| `cpu_heavy_local_bench` | `mapped` | 39 | 0.015575 | sensing-compressibility-pems:13, CFCMT:12, TransitDuet:5, sensing-compressibility-v10k:3, BAPR:3, bapr_v15:2, freqduet:1 |
| `gpu_rl_unmeasured_variant` | `measurement_required` | 16 | 0.006390 | H2Oplus:14, SimpleSAC:2 |
| `scheduler_control_plane` | `measurement_required` | 2 | 0.000799 | sched-hpc-e2e-20260522-214808-175687:1, scheduleurm:1 |

## Interpretation

Global production stability is not closed until the reviewer-facing production population has full strict measured-bucket coverage, or each remaining bucket has its own theorem-grade service certificate.

The strict theorem-facing standard is full measured-bucket coverage of the
declared production population. Representative mappings are diagnostic
unless separately backed by service measurements or equivalence tests.
