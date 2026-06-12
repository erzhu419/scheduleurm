# Natural Live Theorem Trace

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `algorithm` | `theorem_maxweight_v1` |
| `task_count` | 12 |
| `placed_count` | 12 |
| `unplaced_count` | 0 |
| `trace_slot_count` | 12 |
| `theorem_slot_count` | 12 |
| `candidate_count_total` | 68 |
| `alpha0` | 0.000000000 |
| `alpha1` | 0.000000000 |
| `bridge_usable_for_theorem` | true |

## Scope

live-node-probe dry run with measured q01/q11 workload records; queue is not mutated and no process is launched. Uncertified candidate profiles are blocked by the theorem policy during this probe.

## Placements

| Task | Workload | Placement |
|---|---|---|
| `natural-theorem-000` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `natural-theorem-001` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 1, "node": "jtl110gpu2"}` |
| `natural-theorem-002` | `hybrid_rl_resac_ant` | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `natural-theorem-003` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `natural-theorem-004` | `hybrid_rl_resac_ant` | `{"gpu_idx": 1, "node": "node007-direct"}` |
| `natural-theorem-005` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 0, "node": "local"}` |
| `natural-theorem-006` | `hybrid_rl_resac_ant` | `{"gpu_idx": 1, "node": "node007-direct"}` |
| `natural-theorem-007` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 1, "node": "node007-direct"}` |
| `natural-theorem-008` | `hybrid_rl_resac_ant` | `{"gpu_idx": 2, "node": "node007-direct"}` |
| `natural-theorem-009` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 3, "node": "node007-direct"}` |
| `natural-theorem-010` | `hybrid_rl_resac_ant` | `{"gpu_idx": 2, "node": "node007-direct"}` |
| `natural-theorem-011` | `gpu_heavy_jax_matmul` | `{"gpu_idx": 2, "node": "node007-direct"}` |
