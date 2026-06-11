# Module100 Production Theorem Oracle Bridge

This certificate is built from the measured service map and the
reviewer-facing completed-active production population. It is not a
live scheduler dispatch trace.

## Summary

| Quantity | Value |
|---|---:|
| `status` | `SERVICE_MAP_THEOREM_ORACLE_PASS` |
| `completed_active_record_count` | 3437 |
| `mapped_task_count` | 3437 |
| `representative_mapped_task_count` | 0 |
| `unmapped_task_count` | 0 |
| `delta` | 0.000009123 |
| `alpha0` | 0.000000000 |
| `alpha1` | 0.000000000 |
| `usable_for_service_map_oracle_bridge` | true |
| `usable_for_live_scheduler_oracle_trace` | false |

## Interpretation

service-map theorem oracle bridge for the completed-active production population; not a live scheduler dispatch trace and not a raw-history theorem population

The remaining live-oracle requirement is separate:

Capture real scheduler candidate slots with SCHEDULEURM_ORACLE_TRACE_PATH and enrich every candidate with queue_vector, lower_service, penalty_units, and score_semantics=robust_maxweight_lower_service before claiming the live scheduler implements the theorem oracle.
