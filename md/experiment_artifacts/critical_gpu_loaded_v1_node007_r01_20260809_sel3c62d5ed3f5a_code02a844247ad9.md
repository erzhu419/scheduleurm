# Critical GPU Loaded Natural-Completion Campaign

- Status: `INCOMPLETE`
- Node: `node007`
- Wave / split: `1` / `training`

| Scenario | Resident | Target | Ready | Boundary |
|---|---|---|:---:|:---:|
| `llm_after_cnn` | `gpu_cnn_torch_resnet50` | `gpu_llm_distilgpt2` | true | false |
| `rl_after_cnn` | `gpu_cnn_torch_resnet50` | `hybrid_rl_resac_ant` | false | false |
| `cnn_after_rl` | `hybrid_rl_resac_ant` | `gpu_cnn_torch_resnet50` | true | false |
