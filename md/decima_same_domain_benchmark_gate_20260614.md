# Decima Same-Domain Benchmark Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `DECIMA_SAME_DOMAIN_BENCHMARK_EXECUTION_PASS_PERFORMANCE_MIXED` |
| `seed_count` | 3 |
| `completed_pair_count` | 3 |
| `spark_dag_same_domain_benchmark_ready` | true |
| `scheduleurm_gpu_colocation_comparable_ready` | false |

## Rows

| Seed | Dynamic wall | First-frontier wall | Dynamic mean completion | First-frontier mean completion | Completed |
|---:|---:|---:|---:|---:|---:|
| 1 | 226799.0 | 229340.0 | 124963.66666666667 | 126277.0 | true |
| 2 | 629293.0 | 637611.0 | 576350.0 | 560250.3333333334 | true |
| 3 | 378199.0 | 381374.0 | 153105.33333333334 | 155474.33333333334 | true |

## Scope

Same-domain Decima Spark-DAG simulator benchmark.  The closed claim is executable same-domain benchmark coverage, not Decima performance superiority and not GPU co-location or Scheduleurm production-cluster comparability.
