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
| `service_cache_path` | `md/experiment_artifacts/corner_case_service_cache_fresh_eta_20260626.json` |

## Rows

| Scenario | Role | Workload key | Lower service | Lambda | Delta | Eta | Ready |
|---|---|---|---:|---:|---:|---:|---:|
| `node007_empty_small` | `empty_baseline` | `corner_gpu_node007_empty_marginal_cuda` | 490.485 | 392.388 | 98.0969 | 98.0969 | true |
| `node007_30gb_add_small_stable` | `co_location` | `corner_gpu_node007_30gb_resident_marginal_cuda` | 442.512 | 354.01 | 88.5025 | 88.5025 | true |
| `cpu_empty_node001_single` | `empty_baseline` | `corner_cpu_node001_empty_single` | 17.5044 | 14.0035 | 3.50089 | 3.50089 | true |
| `cpu_half_node001_add_single_v2` | `co_location` | `corner_cpu_node001_96worker_resident_single` | 17.3495 | 13.8796 | 3.4699 | 3.4699 | true |
| `cpu_full_node001_add_single_v2` | `co_location` | `corner_cpu_node001_180worker_resident_single` | 17.3817 | 13.9054 | 3.47634 | 3.47634 | true |

## Scope

The gate admits the controlled node007/node001 corner-case rows into a standalone service-cache snapshot and certifies positive row-level capacity slack under a conservative offered-load fraction.  It is not a production-wide queueing theorem, a full fabric-cover proof, or a direct external full-stack SOTA comparison.
