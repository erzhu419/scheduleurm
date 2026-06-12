# Production Queued Theorem Trace

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `algorithm` | `theorem_maxweight_v1` |
| `queued_admissible_count` | 1 |
| `placed_count` | 1 |
| `unplaced_count` | 0 |
| `trace_slot_count` | 1 |
| `theorem_slot_count` | 1 |
| `candidate_count_total` | 3 |
| `bridge_usable_for_theorem` | true |

## Scope

read-only theorem trace over currently queued production tasks; queue is not mutated and no process is launched.

## Placements

| Task | Project | Workload | Placement |
|---|---|---|---|
| `t10409` | `BAPR` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "node007-direct"}` |
