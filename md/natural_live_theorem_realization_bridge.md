# Live Trace Realization Bridge

## Summary

| Quantity | Value |
|---|---:|
| `status` | `DRYRUN_NO_REALIZATION` |
| `trace_slot_count` | 12 |
| `theorem_trace_slot_count` | 12 |
| `matched_task_record_count` | 0 |
| `completed_task_count` | 0 |
| `progress_observation_count` | 0 |
| `usable_for_live_completion_claim` | false |
| `usable_for_live_progress_claim` | false |

## Scope

matches scheduler-emitted trace task ids to queue/archive records. Dry-run theorem traces intentionally have no realized completion claim until the same trace path is produced by actual dispatch.
