# Adaptive Live Integration Probe

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `slot_count` | 6 |
| `trace_origin` | `synthetic_measured_workload_fallback` |
| `selected_count` | 6 |
| `adaptive_sampler_rows` | 6 |
| `regime_detector_rows` | 6 |
| `forced_exploration_count` | 6 |
| `base_oracle_usable` | true |
| `live_scheduler_default_changed` | false |

## Scope

optional adaptive_theorem_maxweight_v1 live-state probe. It uses queued production tasks when present and otherwise falls back to measured synthetic q01/q11 live-node probes. The queue is not mutated, no process is launched, and the default scheduler policy is unchanged.

## Underlying Production Trace

# Natural Live Theorem Trace

## Summary

| Quantity | Value |
|---|---:|
| `pass` | false |
| `algorithm` | `adaptive_theorem_maxweight_v1` |
| `task_count` | 8 |
| `placed_count` | 6 |
| `unplaced_count` | 2 |
| `trace_slot_count` | 6 |
| `theorem_slot_count` | 6 |
| `candidate_count_total` | 23 |
| `alpha0` | 0.000000000 |
| `alpha1` | 0.057561652 |
| `bridge_usable_for_theorem` | true |

## Scope

live-node-probe dry run with measured q01/q11 workload records; queue is not mutated and no process is launched. Uncertified candidate profiles are blocked by the theorem policy during this probe.

## Placements

| Task | Workload | Placement |
|---|---|---|
| `natural-theorem-000` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "jtl110gpu2"}` |
| `natural-theorem-001` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 1, "node": "jtl110gpu"}` |
| `natural-theorem-002` | `hybrid_rl_resac_ant` | `{"gpu_idx": 1, "node": "jtl110gpu2"}` |
| `natural-theorem-003` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 2, "node": "node007-direct"}` |
| `natural-theorem-004` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `natural-theorem-005` | `gpu_heavy_jax_matmul` | `null` |
| `natural-theorem-006` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `natural-theorem-007` | `gpu_heavy_jax_matmul` | `null` |
