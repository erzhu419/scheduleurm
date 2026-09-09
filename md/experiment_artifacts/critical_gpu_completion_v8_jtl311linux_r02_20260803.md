# Critical GPU completion campaign

- Artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_completion_v8_jtl311linux_r02_20260803.json`
- Protocol: `critical_gpu_phase_completion_v8`
- Node: `jtl311linux`
- Wave / role: `2` / `training`
- Status: `PASS`
- Ready rows: `9`
- Capacity boundaries: `0`

| workload | profile | tasks | status | natural completion | checkpoint allocation |
|---|---:|---:|---|---|---|
| `gpu_heavy_jax_matmul` | 1 | 2 | `READY` | True | True |
| `gpu_heavy_jax_matmul` | 3 | 6 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 1 | 2 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 2 | 4 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 1 | 2 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 3 | 6 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 5 | 10 | `READY` | True | True |
| `hybrid_rl_resac_ant` | 2 | 4 | `READY` | True | True |
| `hybrid_rl_resac_ant` | 5 | 10 | `READY` | True | True |
