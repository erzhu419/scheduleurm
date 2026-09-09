# Critical GPU Loaded Action Ledger Merge

- Status: `PASS`
- Actions: `16` / `16`
- Coordinate rows: `32` / `32`
- Legacy index unchanged: `true`

| Action | Node | Resident -> target | Lower-service vector |
|---|---|---|---|
| `gpu_loaded:gpu_b:cnn_after_llm` | `gpu_b` | `gpu_llm_distilgpt2` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 4.5275868970864135, "gpu_llm_distilgpt2": 120.39972693896493}` |
| `gpu_loaded:gpu_b:cnn_after_rl` | `gpu_b` | `hybrid_rl_resac_ant` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 4.1686129440468385, "hybrid_rl_resac_ant": 0.04594463539583408}` |
| `gpu_loaded:gpu_b:llm_after_cnn` | `gpu_b` | `gpu_cnn_torch_resnet50` -> `gpu_llm_distilgpt2` | `{"gpu_cnn_torch_resnet50": 16.652848248271273, "gpu_llm_distilgpt2": 8.123686883312402}` |
| `gpu_loaded:gpu_b:rl_after_cnn` | `gpu_b` | `gpu_cnn_torch_resnet50` -> `hybrid_rl_resac_ant` | `{"gpu_cnn_torch_resnet50": 13.640537644966777, "hybrid_rl_resac_ant": 0.025189095356821043}` |
| `gpu_loaded:gpu_a:cnn_after_llm` | `gpu_a` | `gpu_llm_distilgpt2` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 3.884173435651024, "gpu_llm_distilgpt2": 242.4812265920143}` |
| `gpu_loaded:gpu_a:cnn_after_rl` | `gpu_a` | `hybrid_rl_resac_ant` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 3.782760928593074, "hybrid_rl_resac_ant": 0.10461399553784421}` |
| `gpu_loaded:gpu_a:llm_after_cnn` | `gpu_a` | `gpu_cnn_torch_resnet50` -> `gpu_llm_distilgpt2` | `{"gpu_cnn_torch_resnet50": 20.42386078979414, "gpu_llm_distilgpt2": 8.522174330969264}` |
| `gpu_loaded:gpu_a:rl_after_cnn` | `gpu_a` | `gpu_cnn_torch_resnet50` -> `hybrid_rl_resac_ant` | `{"gpu_cnn_torch_resnet50": 16.205730205198304, "hybrid_rl_resac_ant": 0.051611507341817055}` |
| `gpu_loaded:cpu_a:cnn_after_llm` | `cpu_a` | `gpu_llm_distilgpt2` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 2.8448899943047152, "gpu_llm_distilgpt2": 112.14179600566582}` |
| `gpu_loaded:cpu_a:cnn_after_rl` | `cpu_a` | `hybrid_rl_resac_ant` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 3.0525203769214406, "hybrid_rl_resac_ant": 0.05865404893449546}` |
| `gpu_loaded:cpu_a:llm_after_cnn` | `cpu_a` | `gpu_cnn_torch_resnet50` -> `gpu_llm_distilgpt2` | `{"gpu_cnn_torch_resnet50": 8.502998661578335, "gpu_llm_distilgpt2": 16.38268340170786}` |
| `gpu_loaded:cpu_a:rl_after_cnn` | `cpu_a` | `gpu_cnn_torch_resnet50` -> `hybrid_rl_resac_ant` | `{"gpu_cnn_torch_resnet50": 6.522396354659502, "hybrid_rl_resac_ant": 0.04028755137320146}` |
| `gpu_loaded:node007:cnn_after_llm` | `node007` | `gpu_llm_distilgpt2` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 5.708658253147247, "gpu_llm_distilgpt2": 198.06502366945853}` |
| `gpu_loaded:node007:cnn_after_rl` | `node007` | `hybrid_rl_resac_ant` -> `gpu_cnn_torch_resnet50` | `{"gpu_cnn_torch_resnet50": 5.225748837496235, "hybrid_rl_resac_ant": 0.08437832934557322}` |
| `gpu_loaded:node007:llm_after_cnn` | `node007` | `gpu_cnn_torch_resnet50` -> `gpu_llm_distilgpt2` | `{"gpu_cnn_torch_resnet50": 24.45610399857127, "gpu_llm_distilgpt2": 10.875005500651058}` |
| `gpu_loaded:node007:rl_after_cnn` | `node007` | `gpu_cnn_torch_resnet50` -> `hybrid_rl_resac_ant` | `{"gpu_cnn_torch_resnet50": 18.2455092834486, "hybrid_rl_resac_ant": 0.04566598602291157}` |

The ledger certifies only the four registered, direction-sensitive two-workload trajectories on gpu_a, gpu_b, node007, and cpu_a. The two nominally identical 3080Ti hosts remain separate operational execution classes because their pre-registered equivalence audit failed; their service rows are never pooled. Each action retains its complete two-coordinate lower-service vector. Exact-state cache rows are coordinate views, not independent-action claims, and cannot update the legacy workload/profile index. No claim is made for unmeasured mixtures, hardware, or future workloads.
