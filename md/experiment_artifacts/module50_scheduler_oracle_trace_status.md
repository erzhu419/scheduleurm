# Scheduler Oracle Trace Audit

## Summary

| Quantity | Value |
|---|---:|
| `status` | `NO_TRACE` |
| `trace_slot_count` | 0 |
| `audited_slot_count` | 0 |
| `max_scheduler_score_gap` | 0.000000000 |
| `usable_for_scheduler_score_audit` | false |
| `usable_for_theorem` | false |

## Theorem Blocker

trace file does not exist

## Interpretation

A PASS here certifies that the live scheduler chose the best action under
the recorded scheduler sort key for each traced candidate family.  It is
not by itself a robust MaxWeight theorem certificate unless the trace
rows carry lower-service vectors and declare
`robust_maxweight_lower_service` semantics.
