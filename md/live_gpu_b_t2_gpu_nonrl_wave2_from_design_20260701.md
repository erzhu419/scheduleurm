# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `3`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_mixed_colocation_llm_plus_hybrid_rl_p3` | `gpu_b` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 40.7274 | true | `` |
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cpu_plus_gpu_target_p2` | `gpu_b` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 46.0218 | true | `` |
| `design_gpu_b_gpu_llm_distilgpt2_llm_cpu_resident_cpu_worker_resident_p1` | `gpu_b` | `cpu_resident` | `gpu_llm_distilgpt2` | 0 | false | `stable_rate_not_ready` |
| `design_gpu_b_gpu_llm_distilgpt2_llm_mixed_colocation_cnn_plus_hybrid_rl_p1` | `gpu_b` | `mixed_colocation` | `gpu_llm_distilgpt2` | 427.334 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
