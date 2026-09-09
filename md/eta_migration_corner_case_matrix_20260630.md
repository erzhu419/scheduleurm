# ETA and Migration Corner-Case Coverage Matrix

- Status: `CORNER_CASE_MATRIX_READY`
- Pass: `true`

| Corner case | Category | Gate | Status | Theorem-facing ready |
|---|---|---|---|---:|
| `empty_gpu` | `ETA` | `cross_node_eta_matrix + live_eta_service_cache_v2_gate` | `implemented_live_multi_node_partial` | true |
| `half_loaded_gpu` | `ETA` | `live_marginal_under_load_matrix` | `implemented_live` | true |
| `full_loaded_gpu` | `ETA` | `live_marginal_under_load_full_mixed` | `implemented_live` | true |
| `high_vram_resident` | `ETA` | `live_marginal_under_load_matrix` | `implemented_live` | true |
| `cpu_resident` | `ETA` | `live_marginal_under_load_cpu_idle_nodes` | `implemented_live_representative_and_verify` | true |
| `mixed_cnn_llm` | `ETA` | `live_marginal_under_load_full_mixed` | `implemented_live` | true |
| `mixed_cnn_rl_llm` | `ETA` | `live_marginal_under_load_mixed_cnn_rl_llm_v5` | `implemented_live` | true |
| `rl_periodic_train_eval_eta` | `ETA` | `progress_wrapper stable-cycle-units + live_cycle_jtl110_resac_ant_p2` | `implemented_live` | true |
| `jtl311_cpu_fast_gpu_weak` | `node-aware` | `service_cache_v2` | `implemented_tested` | true |
| `node007_multigpu_packing` | `node-aware` | `live_eta_service_cache_v2_gate` | `implemented_live_partial` | true |
| `node001_node006_cpu_packing` | `node-aware` | `live_marginal_under_load_cpu_idle_nodes` | `implemented_live_representative_and_verify` | true |
| `unknown_env` | `admission` | `service_registry` | `probe_defer_required` | false |
| `description_only_env` | `admission` | `service_registry` | `implemented_tested` | true |
| `missing_tqdm` | `admission` | `service_cache_v2` | `history_fallback_no_theorem_claim` | false |
| `missing_checkpoint` | `migration` | `migration_cost_gate` | `implemented_tested` | true |
| `resume_path_local_only` | `migration` | `migration_cost_gate` | `excluded_until_staged` | false |
| `pure_gpu_migration_25_50_75` | `migration` | `live_checkpoint_migration_cost_gate` | `implemented_live` | true |
| `hybrid_rl_migration_25_50_75` | `migration` | `live_checkpoint_migration_cost_gate` | `implemented_live` | true |
| `pure_cpu_migration_25_50_75` | `migration` | `live_checkpoint_migration_cost_gate` | `implemented_live` | true |
| `port_reberthing` | `OR-generalization` | `or_generalization_benchmarks` | `implemented_simulator` | true |
| `fjsp_reroute` | `OR-generalization` | `or_generalization_benchmarks` | `implemented_simulator` | true |
| `mmrcpsp_mode_switch` | `OR-generalization` | `or_generalization_benchmarks` | `implemented_simulator` | true |

Rows marked requires_live_probe or no_theorem_claim are engineering coverage commitments, not theorem-facing empirical claims.
