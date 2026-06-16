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
| 1 | 228336.0 | 229629.0 | 125700.0 | 126113.0 | true |
| 2 | 627733.0 | 636784.0 | 574660.6666666666 | 558693.6666666666 | true |
| 3 | 376438.0 | 382052.0 | 152633.66666666666 | 155850.66666666666 | true |

## Scope

Same-domain Decima Spark-DAG simulator benchmark.  The closed claim is executable same-domain benchmark coverage, not Decima performance superiority and not GPU co-location or Scheduleurm production-cluster comparability.
