# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `3`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_gpu_cnn_torch_resnet50_cnn_cpu_resident_cpu_worker_resident_p3` | `jtl110gpu2` | `cpu_resident` | `gpu_cnn_torch_resnet50` | 50.1195 | true | `` |
| `design_jtl110gpu2_gpu_cnn_torch_resnet50_cnn_mixed_colocation_llm_plus_hybrid_rl_p2` | `jtl110gpu2` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 40.2283 | true | `` |
| `design_jtl110gpu2_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cpu_plus_gpu_target_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 48.3007 | true | `` |
| `design_jtl110gpu2_gpu_llm_distilgpt2_llm_cpu_resident_cpu_worker_resident_p1` | `jtl110gpu2` | `cpu_resident` | `gpu_llm_distilgpt2` | 0 | false | `stable_rate_not_ready` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
