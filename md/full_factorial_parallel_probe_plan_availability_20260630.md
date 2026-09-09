# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `205`
- Selected rows: `0`
- Max rows per lane: `1`
- Availability audit: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/resource_audit_20260630/corner_case_resource_audit.json`

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
| `gpu_cnn_torch_resnet50` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `high_vram_resident/llm_or_memory_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
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

## Pending Rows Blocked By Current Availability

| Workload | State | Eligible nodes | Availability reasons |
|---|---|---|---|
| `hybrid_rl_resac_ant` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110gpu;jtl110gpu2` | `jtl110gpu:gpu_and_cpu_busy;jtl110gpu2:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_halfcheetah` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110gpu;jtl110gpu2` | `jtl110gpu:gpu_and_cpu_busy;jtl110gpu2:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_hopper` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110gpu;jtl110gpu2` | `jtl110gpu:gpu_and_cpu_busy;jtl110gpu2:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_walker2d` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110gpu;jtl110gpu2` | `jtl110gpu:gpu_and_cpu_busy;jtl110gpu2:cpu_busy_gpu_idle` |
| `hybrid_rl_bapr_ant` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110gpu;jtl110gpu2` | `jtl110gpu:gpu_and_cpu_busy;jtl110gpu2:cpu_busy_gpu_idle` |
| `light_control_local` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `light_control_local` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `cpu_heavy_local_bench` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `cpu_heavy_local_bench` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `freqduet_cpu_ablation_c17_32` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `freqduet_cpu_ablation_c17_32` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `sumo_eval_cpu` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `sumo_eval_cpu` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_ant` | `half_loaded/same_workload_half_capacity` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_ant` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_halfcheetah` | `half_loaded/same_workload_half_capacity` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_halfcheetah` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_hopper` | `half_loaded/same_workload_half_capacity` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_hopper` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_walker2d` | `half_loaded/same_workload_half_capacity` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_walker2d` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_bapr_ant` | `half_loaded/same_workload_half_capacity` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_bapr_ant` | `full_loaded/same_workload_to_capacity_boundary` | `jtl311linux` | `jtl311linux:cpu_busy_gpu_idle` |
| `hybrid_rl_resac_ant` | `half_loaded/same_workload_half_capacity` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_ant` | `full_loaded/same_workload_to_capacity_boundary` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_halfcheetah` | `half_loaded/same_workload_half_capacity` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_halfcheetah` | `full_loaded/same_workload_to_capacity_boundary` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_hopper` | `half_loaded/same_workload_half_capacity` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_hopper` | `full_loaded/same_workload_to_capacity_boundary` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_walker2d` | `half_loaded/same_workload_half_capacity` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_resac_walker2d` | `full_loaded/same_workload_to_capacity_boundary` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_bapr_ant` | `half_loaded/same_workload_half_capacity` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `hybrid_rl_bapr_ant` | `full_loaded/same_workload_to_capacity_boundary` | `node007-direct` | `node007-direct:gpu_and_cpu_busy` |
| `light_control_local` | `cpu_resident/cpu_worker_resident` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `cpu_heavy_local_bench` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `cpu_heavy_local_bench` | `cpu_resident/cpu_worker_resident` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `freqduet_cpu_ablation_c17_32` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `freqduet_cpu_ablation_c17_32` | `cpu_resident/cpu_worker_resident` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `sumo_eval_cpu` | `full_loaded/same_workload_to_capacity_boundary` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |
| `sumo_eval_cpu` | `cpu_resident/cpu_worker_resident` | `jtl110cpu;jtl110cpu2;node007-direct` | `jtl110cpu:missing_audit;jtl110cpu2:missing_audit;node007-direct:gpu_and_cpu_busy` |