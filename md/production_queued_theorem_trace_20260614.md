# Production Queued Theorem Trace

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `algorithm` | `theorem_maxweight_v1` |
| `hard_rule_mode` | `clean_bench` |
| `allow_all_hard_rule_scope` | true |
| `queued_admissible_count` | 1 |
| `placed_count` | 1 |
| `unplaced_count` | 0 |
| `trace_slot_count` | 1 |
| `theorem_slot_count` | 1 |
| `candidate_count_total` | 9 |
| `bridge_usable_for_theorem` | true |

## Scope

read-only theorem trace over currently queued production tasks; queue is not mutated and no process is launched.

## Placements

| Task | Project | Workload | Placement |
|---|---|---|---|
| `t11490` | `BAPR` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 1, "node": "jtl110gpu2"}` |
