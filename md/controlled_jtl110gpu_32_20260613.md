# Controlled Live Theorem Dispatch

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| run_id | `controlled_jtl110gpu_32_20260613` |
| require_node | `jtl110gpu` |
| preferred_node | `` |
| allowed_nodes | `jtl110gpu` |
| max_gpu_util_pct | 95 |
| max_tasks_per_gpu | 16 |
| dispatch_pass_count | 2 |
| task_count | 32 |
| trace_slot_count | 32 |
| theorem_slot_count | 32 |
| candidate_count_total | 56 |
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
| `t11185` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/00` |
| `t11186` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/01` |
| `t11187` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/02` |
| `t11188` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/03` |
| `t11189` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/04` |
| `t11190` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/05` |
| `t11191` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/06` |
| `t11192` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/07` |
| `t11193` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/08` |
| `t11194` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/09` |
| `t11195` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/10` |
| `t11196` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/11` |
| `t11197` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/12` |
| `t11198` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/13` |
| `t11199` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/14` |
| `t11200` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/15` |
| `t11201` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/16` |
| `t11202` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/17` |
| `t11203` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/18` |
| `t11204` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/19` |
| `t11205` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/20` |
| `t11206` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/21` |
| `t11207` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/22` |
| `t11208` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/23` |
| `t11209` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/24` |
| `t11210` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/25` |
| `t11211` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/26` |
| `t11212` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/27` |
| `t11213` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/28` |
| `t11214` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/29` |
| `t11215` | `gpu_heavy_jax_matmul` | `ScheduleurmBench/jax_matmul_size8192/controlled_live_theorem/controlled_jtl110gpu_32_20260613/30` |
| `t11216` | `hybrid_rl_resac_ant` | `ScheduleurmBench/resac_ant_real/controlled_live_theorem/controlled_jtl110gpu_32_20260613/31` |

## Artifacts

- trace: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_32_20260613_trace.jsonl`
- report: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_32_20260613.json`
- realization: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/controlled_jtl110gpu_32_20260613_realization.json`
