# Critical GPU completion campaign

- Artifact: `/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_completion_v3_gpu_b_r01_20260803.json`
- Protocol: `critical_gpu_phase_completion_v3`
- Node: `gpu_b`
- Wave / role: `1` / `training`
- Status: `RUNNING`
- Ready rows: `0`
- Capacity boundaries: `0`

| workload | profile | tasks | status | natural completion | checkpoint allocation |
|---|---:|---:|---|---|---|
| `gpu_heavy_jax_matmul` | 1 | 2 | `READY` | True | True |
| `gpu_heavy_jax_matmul` | 3 | 6 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 1 | 2 | `READY` | True | True |
| `gpu_cnn_torch_resnet50` | 3 | 6 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 1 | 2 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 3 | 6 | `READY` | True | True |
| `gpu_llm_distilgpt2` | 8 | 16 | `READY` | True | True |
