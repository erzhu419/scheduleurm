# Critical GPU Loaded-State Stochastic LCB Gate

- Status: `PASS`
- Node: `jtl110gpu2`
- Ready observations: `52` / `52`

| Scenario | Resident -> target | Upper target s | Lower resident | Lower target | Holdout covered |
|---|---|---:|---:|---:|:---:|
| `cnn_after_llm` | `gpu_llm_distilgpt2` -> `gpu_cnn_torch_resnet50` | 13.2521 | 120.4 | 4.52759 | true |
| `cnn_after_rl` | `hybrid_rl_resac_ant` -> `gpu_cnn_torch_resnet50` | 14.3933 | 0.0459446 | 4.16861 | true |
| `llm_after_cnn` | `gpu_cnn_torch_resnet50` -> `gpu_llm_distilgpt2` | 14.7716 | 16.6528 | 8.12369 | true |
| `rl_after_cnn` | `gpu_cnn_torch_resnet50` -> `hybrid_rl_resac_ant` | 1587.99 | 13.6405 | 0.0251891 | true |

This is a hardware-local certificate for four registered, direction-sensitive two-workload GPU co-location trajectories. Target completion includes initialization, the canonical outer-loop work, checkpoints/final save, and natural exit. Resident service is the counter increment strictly inside the target start/end markers divided by the full overlap duration. One wave-maximum split-conformal margin jointly covers both functionals for all four actions. Every row also hash-verifies the pre-launch nvidia-smi snapshot and rejects undeclared load on any registered GPU, so a hash-bound one-second sidecar covers the complete target interval, bounds aggregate host load and available memory, and rejects high-CPU same-user processes outside the two controlled process groups. Thus same-GPU co-location rows do not silently absorb undeclared host/PCIe contention. It neither populates the legacy workload/profile index nor extrapolates to unmeasured mixtures, hardware, or future workloads.
