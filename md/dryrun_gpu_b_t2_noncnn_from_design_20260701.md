# Live Marginal Under-Load Matrix

- Status: `DRY_RUN_READY`
- Allow launch: `false`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_gpu_b_gpu_heavy_jax_matmul_gpu_matmul_cpu_resident_cpu_worker_resident_p1` | `gpu_b` | `cpu_resident` | `gpu_heavy_jax_matmul` | 0 | false | `dry_run_manifest` |
| `design_gpu_b_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_cnn_plus_hybrid_rl_p1` | `gpu_b` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 0 | false | `dry_run_manifest` |
| `design_gpu_b_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_llm_plus_hybrid_rl_p1` | `gpu_b` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 0 | false | `dry_run_manifest` |
| `design_gpu_b_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_cnn_plus_llm_plus_hybrid_rl_p1` | `gpu_b` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 0 | false | `dry_run_manifest` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
