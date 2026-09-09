# Live Marginal Under-Load Matrix

- Status: `DRY_RUN_READY`
- Allow launch: `false`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_cnn_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 0 | false | `dry_run_manifest` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_llm_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 0 | false | `dry_run_manifest` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_cnn_plus_llm_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 0 | false | `dry_run_manifest` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
