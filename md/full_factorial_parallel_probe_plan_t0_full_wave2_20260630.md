# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `24`
- Selected rows: `5`
- Max rows per lane: `1`

## Resource Lock Policy

At most one active probe per physical node.  A node-level lock covers both CPU and GPU resources, so CPU resident probes never overlap with GPU ETA probes on the same host.  Homogeneous equivalent nodes are used only for equivalence validation unless explicitly selected as a spillover lane.

## Lanes

| Lane | Node lock | Resource lock | Selected | Commands |
|---|---|---|---:|---:|
| `gpu_jtl110_representative` | `gpu_a` | `gpu_a:gpu` | 0 | 0 |
| `gpu_jtl110_spillover` | `gpu_b` | `gpu_b:gpu` | 1 | 1 |
| `gpu_jtl311_cpu_fast` | `cpu_a` | `cpu_a:gpu` | 1 | 1 |
| `gpu_node007_direct` | `node007-direct` | `node007-direct:gpu` | 1 | 1 |
| `cpu_hpc_192c_representative` | `node001` | `node001:cpu` | 1 | 1 |
| `cpu_hpc_192c_spillover_node002` | `node002` | `node002:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node004` | `node004` | `node004:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node005` | `node005` | `node005:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node006` | `node006` | `node006:cpu` | 0 | 0 |
| `cpu_hpc_192c_equivalence` | `node003` | `node003:cpu` | 0 | 0 |
| `cpu_jtl110_representative` | `windows_a` | `windows_a:cpu` | 0 | 0 |
| `cpu_jtl110_spillover` | `windows_b` | `windows_b:cpu` | 1 | 1 |
| `cpu_node007_jtl110_family` | `node007-direct` | `node007-direct:cpu` | 0 | 0 |
| `cpu_jtl311_host` | `cpu_a` | `cpu_a:cpu` | 0 | 0 |

## Selected Rows

| Lane | Workload | Node | State | Missing profiles | Command |
|---|---|---|---|---|---|
| `gpu_jtl110_spillover` | `gpu_cnn_torch_resnet50` | `gpu_b` | `full_loaded` | `5,6,8` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/parallel_probe_assignments_20260630/ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl110_spillover_gpu_3080ti_12gb_dual_q01_low_cpu_high_gpu_gpu_cnn_torch_resnet50_cnn_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes gpu_b --workloads gpu_cnn_torch_resnet50 --max-rows 1 --run-prefix ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl110_spillover_gpu_b_full_loaded_gpu_cnn_torch_resnet50 --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl110_spillover_gpu_b_full_loaded_gpu_cnn_torch_resnet50.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl110_spillover_gpu_b_full_loaded_gpu_cnn_torch_resnet50.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl110_spillover_gpu_b_full_loaded_gpu_cnn_torch_resnet50.md` |
| `gpu_jtl311_cpu_fast` | `hybrid_rl_bapr_ant` | `cpu_a` | `full_loaded` | `1,2,3,4,5,6` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/parallel_probe_assignments_20260630/ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl311_cpu_fast_gpu_rtx2080_8gb_dual_cpu_fast_q11_high_cpu_high_gpu_hybrid_rl_bapr_ant_bapr_ant_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes cpu_a --workloads hybrid_rl_bapr_ant --max-rows 1 --run-prefix ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_hybrid_rl_bapr_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_hybrid_rl_bapr_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_hybrid_rl_bapr_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_hybrid_rl_bapr_ant.md` |
| `gpu_node007_direct` | `hybrid_rl_bapr_ant` | `node007-direct` | `full_loaded` | `1,2,3,4,5,6` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --design-csv /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/parallel_probe_assignments_20260630/ff_parallel_eta_t0_full_wave2_20260630_gpu_node007_direct_gpu_node007_4x11gb_q11_high_cpu_high_gpu_hybrid_rl_bapr_ant_bapr_ant_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes node007-direct --workloads hybrid_rl_bapr_ant --max-rows 1 --run-prefix ff_parallel_eta_t0_full_wave2_20260630_gpu_node007_direct_node007-direct_full_loaded_hybrid_rl_bapr_ant --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_node007_direct_node007-direct_full_loaded_hybrid_rl_bapr_ant.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_node007_direct_node007-direct_full_loaded_hybrid_rl_bapr_ant.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_gpu_node007_direct_node007-direct_full_loaded_hybrid_rl_bapr_ant.md` |
| `cpu_hpc_192c_representative` | `sumo_eval_cpu` | `node001` | `full_loaded` | `1,2,4,8,16,32,64,96,128` | `python3 -m algorithm.experiments.full_factorial_cpu_eta_probe_runner --allow-launch --design-csv /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/parallel_probe_assignments_20260630/ff_parallel_eta_t0_full_wave2_20260630_cpu_hpc_192c_representative_cpu_hpc_192c_q10_high_cpu_low_gpu_sumo_eval_cpu_sumo_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes node001 --workloads sumo_eval_cpu --max-rows 1 --run-prefix ff_parallel_eta_t0_full_wave2_20260630_cpu_hpc_192c_representative_node001_full_loaded_sumo_eval_cpu --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_hpc_192c_representative_node001_full_loaded_sumo_eval_cpu.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_hpc_192c_representative_node001_full_loaded_sumo_eval_cpu.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_hpc_192c_representative_node001_full_loaded_sumo_eval_cpu.md` |
| `cpu_jtl110_spillover` | `cpu_heavy_local_bench` | `windows_b` | `full_loaded` | `1,2,4,8,16,32,64,96,128,180,192` | `python3 -m algorithm.experiments.full_factorial_cpu_eta_probe_runner --allow-launch --design-csv /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/parallel_probe_assignments_20260630/ff_parallel_eta_t0_full_wave2_20260630_cpu_jtl110_spillover_cpu_jtl110_128c_q10_high_cpu_low_gpu_cpu_heavy_local_bench_cpu_full_loaded_same_workload_to_capacity_boundary.csv --tier T0_core_curve --resource-state full_loaded --nodes windows_b --workloads cpu_heavy_local_bench --max-rows 1 --run-prefix ff_parallel_eta_t0_full_wave2_20260630_cpu_jtl110_spillover_windows_b_full_loaded_cpu_heavy_local_bench --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_jtl110_spillover_windows_b_full_loaded_cpu_heavy_local_bench.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_jtl110_spillover_windows_b_full_loaded_cpu_heavy_local_bench.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_wave2_20260630_cpu_jtl110_spillover_windows_b_full_loaded_cpu_heavy_local_bench.md` |

## Claim Boundary

This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service cache only after their runner writes task-native tqdm/progress stable-rate summaries.
