# Global Theorem Dispatcher Prototype Gate

This gate passes the scoped prototype certificate. It does not claim the live scheduler uses global batch dispatch by default.

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `GLOBAL_THEOREM_DISPATCHER_PROTOTYPE_PASS_NOT_LIVE_DEFAULT` |
| `global_action_dispatch_ready` | true |
| `live_scheduler_default_global_dispatcher_ready` | false |
| `candidate_configuration_count` | 3 |
| `oracle_gap_alpha0` | 0.0 |
| `oracle_gap_alpha1` | 0.0 |
| `score_semantics` | `robust_maxweight_lower_service` |
| `fallback_legacy_rows_excluded` | true |

## Scope

Prototype-only theorem dispatcher.  It enumerates bounded global configurations from theorem-ready rows and excludes legacy fallback rows.
