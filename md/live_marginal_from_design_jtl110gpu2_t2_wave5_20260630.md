# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `1`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_cnn_plus_llm_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 318.478 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
