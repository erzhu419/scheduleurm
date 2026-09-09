# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `4`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_cpu_resident_cpu_worker_resident_p2` | `gpu_b` | `cpu_resident` | `gpu_cnn_torch_resnet50` | 51.2493 | true | `` |
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cnn_plus_hybrid_rl_p3` | `gpu_b` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 47.6171 | true | `` |
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_mixed_colocation_llm_plus_hybrid_rl_p1` | `gpu_b` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 40.3455 | true | `` |
| `design_gpu_b_gpu_cnn_torch_resnet50_cnn_mixed_colocation_cnn_plus_llm_plus_hybrid_rl_p2` | `gpu_b` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 39.9875 | true | `` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
