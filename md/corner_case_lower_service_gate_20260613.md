# Corner-Case Lower-Service Gate

This artifact admits only stable canonical corner-case probes into a standalone service cache. It does not modify the default replay cache.

| Quantity | Value |
|---|---:|
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `service_cache_record_count` | 5 |
| `corner_case_lower_service_capacity_ready` | true |
| `co_location_lower_service_capacity_ready` | true |
| `cache_roundtrip_ready` | true |
| `offered_load_fraction` | 0.8 |
| `service_cache_path` | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/corner_case_service_cache_20260613.json` |

## Rows

| Scenario | Role | Workload key | Lower service | Lambda | Delta | Eta | Ready |
|---|---|---|---:|---:|---:|---:|---:|
| `node007_empty_small` | `empty_baseline` | `corner_gpu_node007_empty_marginal_cuda` | 8534.69 | 6827.76 | 1706.94 | 1706.94 | true |
| `node007_30gb_add_small_stable` | `co_location` | `corner_gpu_node007_30gb_resident_marginal_cuda` | 9344.57 | 7475.66 | 1868.91 | 1868.91 | true |
| `cpu_empty_node001_single` | `empty_baseline` | `corner_cpu_node001_empty_single` | 20.4422 | 16.3537 | 4.08844 | 4.08844 | true |
| `cpu_half_node001_add_single_v2` | `co_location` | `corner_cpu_node001_96worker_resident_single` | 22.3118 | 17.8495 | 4.46237 | 4.46237 | true |
| `cpu_full_node001_add_single_v2` | `co_location` | `corner_cpu_node001_180worker_resident_single` | 20.4698 | 16.3759 | 4.09396 | 4.09396 | true |

## Scope

The gate admits the controlled node007/node001 corner-case rows into a standalone service-cache snapshot and certifies positive row-level capacity slack under a conservative offered-load fraction.  It is not a production-wide queueing theorem, a full fabric-cover proof, or a direct external full-stack SOTA comparison.
