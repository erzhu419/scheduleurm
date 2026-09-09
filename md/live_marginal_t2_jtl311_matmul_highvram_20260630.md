# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `1`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl311linux_gpu_heavy_jax_matmul_gpu_matmul_high_vram_resident_llm_or_memory_resident_p1` | `jtl311linux` | `high_vram_resident` | `gpu_heavy_jax_matmul` | 805.745 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
