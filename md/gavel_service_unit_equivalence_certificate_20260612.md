# Gavel Service-Unit Equivalence Certificate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `trace_schema_compatibility_ready` | true |
| `exact_arrival_times_ready` | false |
| `throughput_seed_ready` | true |
| `gavel_native_bounded_microbaseline_ready` | true |
| `gavel_service_unit_equivalence_ready` | false |
| `direct_full_stack_same_workload_ready` | false |

## Rows

| Taskset | Jobs | Native rows | Total units | Resource counts | Max arrival jitter s | Throughput profiles | Ready |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | 48 | 48 | 115200 | [2] | 0.047 | 8 | true |
| `q11_cpu_gpu_coupled` | 160 | 160 | 12800 | [1] | 0.159 | 9 | true |

## Remaining Blocker

Gavel native trace rows use Gavel template job types and simulator throughput tables, while Scheduleurm theorem rows use measured aggregate lower-service rates from the Scheduleurm service cache.  The trace schema, job counts, arrival times, resource counts, and total_units are aligned, but no calibration experiment currently proves that one Gavel simulator step has the same service-unit meaning as one Scheduleurm measured service unit.

## Scope

Certifies bounded same-trace Gavel adapter compatibility and records the remaining service-unit blocker.  It intentionally does not promote Gavel native simulator rows to full-stack SOTA superiority.
