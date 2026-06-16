# Controlled Live Theorem Dispatch

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| run_id | `controlled_jtl110gpu_pilot_4_resolved_20260613` |
| require_node | `jtl110gpu` |
| preferred_node | `` |
| allowed_nodes | `jtl110gpu` |
| max_gpu_util_pct | 25 |
| dispatch_pass_count | 1 |
| task_count | 4 |
| trace_slot_count | 4 |
| theorem_slot_count | 4 |
| candidate_count_total | 8 |
| dispatch_returncode | 0 |
| wait_returncode | 0 |
| alpha0 | 0.000000000 |
| alpha1 | 0.000000000 |
| realization_status | `LIVE_COMPLETION_REALIZATION_PASS` |
| live_completion_claim | true |
| live_progress_claim | true |

## Scope

controlled local live dispatch smoke using scheduler.py dispatch --algorithm theorem_maxweight_v1. The scheduler queue is mutated, tasks are actually launched, candidate-family traces are emitted, and realization is matched against queue/archive records. This is not yet the full global q01/q11 production experiment.

## Submitted Tasks

| task | workload | signature |
|---|---|---|
| `t11181` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_pilot_4_resolved_20260613/00` |
| `t11182` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_pilot_4_resolved_20260613/01` |
| `t11183` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_pilot_4_resolved_20260613/02` |
| `t11184` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_pilot_4_resolved_20260613/03` |

## Artifacts

- trace: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_pilot_4_resolved_20260613_trace.jsonl`
- report: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_pilot_4_resolved_20260613.json`
- realization: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_pilot_4_resolved_20260613_realization.json`
