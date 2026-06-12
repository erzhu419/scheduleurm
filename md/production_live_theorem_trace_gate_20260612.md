# Production-Wide Live Theorem Trace Gate

## Summary

| Quantity | Value |
|---|---:|
| `status` | `PRODUCTION_WIDE_LIVE_TRACE_PASS` |
| `active_count` | 35 |
| `queued_count` | 1 |
| `production_queued_count` | 1 |
| `admissible_production_queued_count` | 1 |
| `trace_slot_count` | 1 |
| `trace_theorem_slot_count` | 1 |
| `trace_oracle_usable` | true |
| `production_wide_live_trace_closed` | true |

## Scope

This is a gate for production-wide live theorem trace claims. Controlled ScheduleurmBench traces and completed-history service-map bridges are not counted as production-wide live traces.  If no admissible queued production task exists, the correct status is WAIT, not a theorem-grade production dispatch claim.

## Admissible Queued Production

| Task | Project | Workload | Profiles | Submitted |
|---|---|---|---|---:|
| `t10409` | `BAPR` | `hybrid_rl_resac_ant` | `1,2,3,4,5,6,7,8,9` | 1781212957.564 |
