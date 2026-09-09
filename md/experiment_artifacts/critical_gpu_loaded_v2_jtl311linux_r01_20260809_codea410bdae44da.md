# Critical GPU Loaded Natural-Completion Campaign

- Status: `PASS`
- Node: `jtl311linux`
- Wave / split: `1` / `training`

| Scenario | Resident | Target | Ready | Boundary |
|---|---|---|:---:|:---:|
| `cnn_after_llm` | `gpu_llm_distilgpt2` | `gpu_cnn_torch_resnet50` | true | false |
| `llm_after_cnn` | `gpu_cnn_torch_resnet50` | `gpu_llm_distilgpt2` | true | false |
| `rl_after_cnn` | `gpu_cnn_torch_resnet50` | `hybrid_rl_resac_ant` | true | false |
| `cnn_after_rl` | `hybrid_rl_resac_ant` | `gpu_cnn_torch_resnet50` | true | false |
