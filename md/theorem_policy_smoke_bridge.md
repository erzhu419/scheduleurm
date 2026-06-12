# Module52 Theorem Oracle Trace Bridge

## Summary

| Quantity | Value |
|---|---:|
| `status` | `NOT_THEOREM_TRACE` |
| `trace_slot_count` | 8 |
| `converted_slot_count` | 0 |
| `blocker_count` | 8 |
| `alpha0` | NA |
| `alpha1` | NA |
| `usable_for_theorem` | false |

## Blockers

| Slot | Reason | Detail |
|---|---|---|
| `slot:dryrun-oracle-000:1781183108.401978:1781183108409` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-001:1781183108.402978:1781183109184` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-002:1781183108.403978:1781183109375` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-003:1781183108.404978:1781183109582` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-004:1781183108.405978:1781183109772` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-005:1781183108.4069781:1781183109928` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-006:1781183108.407978:1781183110109` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |
| `slot:dryrun-oracle-007:1781183108.408978:1781183110300` | `score_semantics_not_robust_maxweight_lower_service` | `scheduler_sort_key_minimization` |

## Interpretation

This bridge only accepts robust MaxWeight lower-service trace slots.  A
scheduler-sort-key trace can pass Module50 and still fail here; that is
the intended separation between implementation audit and theorem
alpha0/alpha1 calibration.
