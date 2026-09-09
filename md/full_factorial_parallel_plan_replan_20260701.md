# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `188`
- Selected rows: `10`
- Max rows per lane: `4`
- Availability audit: `not provided`

## Resource Lock Policy

At most one active probe per physical node.  A node-level lock covers both CPU and GPU resources, so CPU resident probes never overlap with GPU ETA probes on the same host.  Homogeneous equivalent nodes are used only for equivalence validation unless explicitly selected as a spillover lane.

## Lanes

| Lane | Node lock | Resource lock | Selected | Commands |
|---|---|---|---:|---:|
| `gpu_jtl110_representative` | `jtl110gpu` | `jtl110gpu:gpu` | 2 | 2 |
| `gpu_jtl110_spillover` | `jtl110gpu2` | `jtl110gpu2:gpu` | 0 | 0 |
| `gpu_jtl311_cpu_fast` | `jtl311linux` | `jtl311linux:gpu` | 4 | 4 |
| `gpu_node007_direct` | `node007-direct` | `node007-direct:gpu` | 4 | 4 |
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
| `gpu_jtl110_representative` | `hybrid_rl_bapr_ant` | `jtl110gpu` | `full_loaded` | `4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl110_representative_gpu_3080ti_12gb_dual_q11_high_cpu_high_gpu_hybrid_rl_bapr_ant_bapr_ant_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes jtl110gpu --workloads hybrid_rl_bapr_ant --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_bapr_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_bapr_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_bapr_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_bapr_ant.md` |
| `gpu_jtl110_representative` | `hybrid_rl_resac_ant` | `jtl110gpu` | `full_loaded` | `6` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl110_representative_gpu_3080ti_12gb_dual_q11_high_cpu_high_gpu_hybrid_rl_resac_ant_ant_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes jtl110gpu --workloads hybrid_rl_resac_ant --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_resac_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_resac_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_resac_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl110_representative_jtl110gpu_full_loaded_hybrid_rl_resac_ant.md` |
| `gpu_jtl311_cpu_fast` | `hybrid_rl_bapr_ant` | `jtl311linux` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_gpu_rtx2080_8gb_dual_cpu_fast_q11_high_cpu_high_gpu_hybrid_rl_bapr_ant_bapr_ant_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes jtl311linux --workloads hybrid_rl_bapr_ant --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_bapr_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_bapr_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_bapr_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_bapr_ant.md` |
| `gpu_jtl311_cpu_fast` | `hybrid_rl_resac_ant` | `jtl311linux` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_gpu_rtx2080_8gb_dual_cpu_fast_q11_high_cpu_high_gpu_hybrid_rl_resac_ant_ant_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes jtl311linux --workloads hybrid_rl_resac_ant --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_ant.md` |
| `gpu_jtl311_cpu_fast` | `hybrid_rl_resac_halfcheetah` | `jtl311linux` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_gpu_rtx2080_8gb_dual_cpu_fast_q11_high_cpu_high_gpu_hybrid_rl_resac_halfcheetah_halfcheetah_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes jtl311linux --workloads hybrid_rl_resac_halfcheetah --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_halfcheetah --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_halfcheetah.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_halfcheetah.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_halfcheetah.md` |
| `gpu_jtl311_cpu_fast` | `hybrid_rl_resac_hopper` | `jtl311linux` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_gpu_rtx2080_8gb_dual_cpu_fast_q11_high_cpu_high_gpu_hybrid_rl_resac_hopper_hopper_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes jtl311linux --workloads hybrid_rl_resac_hopper --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_hopper --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_hopper.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_hopper.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_jtl311_cpu_fast_jtl311linux_half_loaded_hybrid_rl_resac_hopper.md` |
| `gpu_node007_direct` | `hybrid_rl_resac_ant` | `node007-direct` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_node007_direct_gpu_node007_4x11gb_q11_high_cpu_high_gpu_hybrid_rl_resac_ant_ant_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes node007-direct --workloads hybrid_rl_resac_ant --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_ant.md` |
| `gpu_node007_direct` | `hybrid_rl_resac_halfcheetah` | `node007-direct` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_node007_direct_gpu_node007_4x11gb_q11_high_cpu_high_gpu_hybrid_rl_resac_halfcheetah_halfcheetah_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes node007-direct --workloads hybrid_rl_resac_halfcheetah --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_halfcheetah --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_halfcheetah.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_halfcheetah.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_halfcheetah.md` |
| `gpu_node007_direct` | `hybrid_rl_resac_hopper` | `node007-direct` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_node007_direct_gpu_node007_4x11gb_q11_high_cpu_high_gpu_hybrid_rl_resac_hopper_hopper_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes node007-direct --workloads hybrid_rl_resac_hopper --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_hopper --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_hopper.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_hopper.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_hopper.md` |
| `gpu_node007_direct` | `hybrid_rl_resac_walker2d` | `node007-direct` | `half_loaded` | `2,3,4` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv md/experiment_artifacts/full_factorial_parallel_assignments_replan_20260701/ff_parallel_eta_replan_20260701_gpu_node007_direct_gpu_node007_4x11gb_q11_high_cpu_high_gpu_hybrid_rl_resac_walker2d_walker2d_half_loaded_same_workload_half_capacity.csv --tier T0_core_curve --resource-state half_loaded --nodes node007-direct --workloads hybrid_rl_resac_walker2d --max-rows 1 --run-prefix ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_walker2d --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_walker2d.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_walker2d.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_replan_20260701_gpu_node007_direct_node007-direct_half_loaded_hybrid_rl_resac_walker2d.md` |

## Claim Boundary

This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service cache only after their runner writes task-native tqdm/progress stable-rate summaries.


## Pending Rows Requiring Dedicated Harness

| Workload | State | Eligible nodes | Reason |
|---|---|---|---|
| `gpu_cnn_torch_resnet50` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `cpu_resident/cpu_worker_resident` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cpu_plus_gpu_target` | `jtl110gpu;jtl110gpu2` | `needs resident/mixed-load harness` |
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
| `light_control_local` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_cnn_torch_resnet50` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_llm_distilgpt2` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `gpu_heavy_jax_matmul` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `cpu_heavy_local_bench` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `freqduet_cpu_ablation_c17_32` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `sumo_eval_cpu` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cnn_plus_llm_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_ant` | `mixed_colocation/cpu_plus_gpu_target` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `high_vram_resident/llm_or_memory_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `cpu_resident/cpu_worker_resident` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cnn_plus_llm` | `jtl311linux` | `needs resident/mixed-load harness` |
| `hybrid_rl_resac_halfcheetah` | `mixed_colocation/cnn_plus_hybrid_rl` | `jtl311linux` | `needs resident/mixed-load harness` |