# Scheduler Oracle Trace Audit

## Summary

| Quantity | Value |
|---|---:|
| `status` | `SCHEDULER_SCORE_PASS` |
| `trace_slot_count` | 2 |
| `audited_slot_count` | 2 |
| `max_scheduler_score_gap` | 0.000000000 |
| `usable_for_scheduler_score_audit` | true |
| `usable_for_theorem` | false |

## Theorem Blocker

trace uses scheduler sort-key semantics, not robust MaxWeight lower-service semantics

## Audited Slots

| Slot | Candidates | Selected | Best | Gap | Semantics |
|---|---:|---|---|---:|---|
| `slot:t9981:1781145424.92721:1781145677613` | 1 | `node=local|cpu` | `node=local|cpu` | 0.000000000 | `scheduler_sort_key_minimization` |
| `slot:t9983:1781145424.92721:1781145827811` | 1 | `node=local|cpu` | `node=local|cpu` | 0.000000000 | `scheduler_sort_key_minimization` |

## Interpretation

A PASS here certifies that the live scheduler chose the best action under
the recorded scheduler sort key for each traced candidate family.  It is
not by itself a robust MaxWeight theorem certificate unless the trace
rows carry lower-service vectors and declare
`robust_maxweight_lower_service` semantics.
