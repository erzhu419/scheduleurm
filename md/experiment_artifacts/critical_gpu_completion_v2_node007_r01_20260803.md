# Critical GPU completion campaign

- Artifact: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_completion_v2_node007_r01_20260803.json`
- Protocol: `critical_gpu_phase_completion_v2`
- Node: `node007`
- Wave / role: `1` / `training`
- Status: `INCOMPLETE`
- Ready rows: `7`
- Capacity boundaries: `0`

| workload | profile | tasks | status | natural completion | checkpoint allocation |
|---|---:|---:|---|---|---|
| `gpu_heavy_jax_matmul` | 1 | 4 | `READY` | True | True |
| `gpu_heavy_jax_matmul` | 3 | 12 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 1 | 4 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 3 | 12 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 1 | 4 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 3 | 12 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 10 | 40 | `FAILED_EXCEPTION` | False | False |
| `hybrid_rl_resac_ant` | 2 | 8 | `READY` | True | True |
| `hybrid_rl_resac_ant` | 5 | 20 | `FAILED` | False | True |
