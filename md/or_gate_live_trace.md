# OR Gate: Longer Live Theorem Trace

| Quantity | Value |
|---|---:|
| pass | true |
| status | `LIVE_SCHEDULER_THEOREM_ORACLE_PASS` |
| trace_slot_count | 128 |
| min_required_slots | 96 |
| candidate_count_total | 767 |
| usable_for_live_scheduler_oracle_trace | true |
| trace_origin | `synthetic_dryrun_live_node_probe` |

## Scope

candidate-family oracle audit over live node-state traces. When trace_origin is synthetic_dryrun_live_node_probe, the scheduler queue was not modified and no task was launched; it certifies placement candidate semantics on current probed resources, not actual dispatch completion.

## Dry-Run Probe

| Quantity | Value |
|---|---:|
| placed_count | 128 |
| unplaced_count | 0 |
| trace_slot_count | 128 |
| candidate_count_total | 767 |
| algorithm | `sweetspot_v1` |

synthetic task records against current live node probe; queue is not modified and no task is launched
