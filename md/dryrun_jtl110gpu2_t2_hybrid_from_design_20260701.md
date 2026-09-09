# Live Marginal Under-Load Matrix

- Status: `DRY_RUN_READY`
- Allow launch: `false`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_hybrid_rl_resac_ant_ant_high_vram_resident_llm_or_memory_resident_p1` | `jtl110gpu2` | `high_vram_resident` | `hybrid_rl_resac_ant` | 0 | false | `dry_run_manifest` |
| `design_jtl110gpu2_hybrid_rl_resac_ant_ant_cpu_resident_cpu_worker_resident_p1` | `jtl110gpu2` | `cpu_resident` | `hybrid_rl_resac_ant` | 0 | false | `dry_run_manifest` |
| `design_jtl110gpu2_hybrid_rl_resac_ant_ant_mixed_colocation_cnn_plus_llm_p1` | `jtl110gpu2` | `mixed_colocation` | `hybrid_rl_resac_ant` | 0 | false | `dry_run_manifest` |
| `design_jtl110gpu2_hybrid_rl_resac_ant_ant_mixed_colocation_cnn_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `hybrid_rl_resac_ant` | 0 | false | `dry_run_manifest` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
