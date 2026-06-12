# Controlled Live Theorem Dispatch

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| run_id | `live_theorem_dispatch_bounded_portfolio_20260611_003` |
| require_node | `` |
| preferred_node | `` |
| allowed_nodes | `local,node007-direct` |
| max_gpu_util_pct | 20 |
| dispatch_pass_count | 2 |
| task_count | 6 |
| trace_slot_count | 7 |
| theorem_slot_count | 7 |
| candidate_count_total | 13 |
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
| `t10315` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/00` |
| `t10316` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/01` |
| `t10317` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/02` |
| `t10318` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/03` |
| `t10319` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/04` |
| `t10320` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/live_theorem_dispatch_bounded_portfolio_20260611_003/05` |

## Artifacts

- trace: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_trace.jsonl`
- report: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003.json`
- realization: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_realization.json`
