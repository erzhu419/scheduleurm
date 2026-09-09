# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `203`
- Selected rows: `0`
- Max rows per lane: `1`

## Resource Lock Policy

At most one active probe per physical node.  A node-level lock covers both CPU and GPU resources, so CPU resident probes never overlap with GPU ETA probes on the same host.  Homogeneous equivalent nodes are used only for equivalence validation unless explicitly selected as a spillover lane.

## Lanes

| Lane | Node lock | Resource lock | Selected | Commands |
|---|---|---|---:|---:|
| `gpu_jtl110_representative` | `jtl110gpu` | `jtl110gpu:gpu` | 0 | 0 |
| `gpu_jtl110_spillover` | `jtl110gpu2` | `jtl110gpu2:gpu` | 0 | 0 |
| `gpu_jtl311_cpu_fast` | `jtl311linux` | `jtl311linux:gpu` | 0 | 0 |
| `gpu_node007_direct` | `node007-direct` | `node007-direct:gpu` | 0 | 0 |
| `cpu_hpc_192c_representative` | `node001` | `node001:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node002` | `node002` | `node002:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node004` | `node004` | `node004:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node005` | `node005` | `node005:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node006` | `node006` | `node006:cpu` | 0 | 0 |
| `cpu_hpc_192c_equivalence` | `node003` | `node003:cpu` | 0 | 0 |
| `cpu_jtl110_representative` | `jtl110cpu` | `jtl110cpu:cpu` | 0 | 0 |
| `cpu_jtl110_spillover` | `jtl110cpu2` | `jtl110cpu2:cpu` | 0 | 0 |
| `cpu_node007_jtl110_family` | `node007-direct` | `node007-direct:cpu` | 0 | 0 |
| `cpu_jtl311_host` | `jtl311linux` | `jtl311linux:cpu` | 0 | 0 |

## Selected Rows

| Lane | Workload | Node | State | Missing profiles | Command |
|---|---|---|---|---|---|

## Claim Boundary

This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service cache only after their runner writes task-native tqdm/progress stable-rate summaries.


## Pending Rows Requiring Dedicated Harness

| Workload | State | Eligible nodes | Reason |
|---|---|---|---|
| `gpu_cnn_torch_resnet50` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_hopper` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_walker2d` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `mixed_colocation/cnn_plus_llm` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `hybrid_rl_bapr_ant` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |