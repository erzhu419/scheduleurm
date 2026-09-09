# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `4`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cpu_plus_gpu_target_p3` | `jtl110gpu2` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 54.3454 | true | `` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_llm_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 451.388 | true | `` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_cnn_plus_llm_plus_hybrid_rl_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 433.857 | true | `` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_mixed_colocation_cpu_plus_gpu_target_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_llm_distilgpt2` | 0 | false | `stable_rate_not_ready` |
| `design_jtl110gpu2_gpu_heavy_jax_matmul_gpu_matmul_mixed_colocation_cpu_plus_gpu_target_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_heavy_jax_matmul` | 426.32 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
