# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `3`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl311linux_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cnn_plus_llm_p1` | `jtl311linux` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 6.05962 | true | `` |
| `design_jtl311linux_gpu_llm_distilgpt2_llm_mixed_colocation_cnn_plus_llm_p1` | `jtl311linux` | `mixed_colocation` | `gpu_llm_distilgpt2` | 215.256 | true | `` |
| `design_jtl311linux_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_cnn_plus_llm_p1` | `jtl311linux` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 205.632 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
