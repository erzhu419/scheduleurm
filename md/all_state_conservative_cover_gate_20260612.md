# All-State Conservative Fabric-Cover Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `admitted_measured_workload_count` | 113 |
| `all_state_safety_cover_ready` | true |
| `positive_service_all_state_cover_ready` | false |
| `all_state_stability_for_unknown_positive_arrivals_ready` | false |

## Partition

| Region | Contract |
|---|---|
| `measured_admitted` | exact positive service-cache profile; identity projection; rho=0 |
| `unknown_unmeasured` | probe/defer action; zero theorem service; not counted as positive-load theorem population |

## Unknown Catch-All

`{"action_id": "unknown_probe_or_defer", "lower_service": 0.0, "projection": "catch_all_probe_action", "rho": 0.0, "theorem_population": "excluded_until_measured"}`

## Scope

Covers every scheduler-visible future state conservatively by partitioning it into admitted measured states or an unknown probe/defer action.  The unknown action has zero theorem service, so this is an all-state safety cover, not a positive-service all-state fabric theorem.
