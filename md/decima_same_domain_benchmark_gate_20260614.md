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
| 1 | 227260.0 | 229923.0 | 124764.33333333333 | 125731.0 | true |
| 2 | 631002.0 | 641520.0 | 578389.6666666666 | 563389.6666666666 | true |
| 3 | 376798.0 | 381165.0 | 153024.0 | 155810.0 | true |

## Scope

Same-domain Decima Spark-DAG simulator benchmark.  The closed claim is executable same-domain benchmark coverage, not Decima performance superiority and not GPU co-location or Scheduleurm production-cluster comparability.
