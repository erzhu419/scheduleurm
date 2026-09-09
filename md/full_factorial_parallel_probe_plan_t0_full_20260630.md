# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `34`
- Selected rows: `5`
- Max rows per lane: `1`

## Resource Lock Policy

At most one active probe per physical node.  A node-level lock covers both CPU and GPU resources, so CPU resident probes never overlap with GPU ETA probes on the same host.  Homogeneous equivalent nodes are used only for equivalence validation unless explicitly selected as a spillover lane.

## Lanes

| Lane | Node lock | Resource lock | Selected | Commands |
|---|---|---|---:|---:|
| `gpu_jtl110_representative` | `gpu_a` | `gpu_a:gpu` | 1 | 1 |
| `gpu_jtl311_cpu_fast` | `cpu_a` | `cpu_a:gpu` | 1 | 1 |
| `gpu_node007_direct` | `node007-direct` | `node007-direct:gpu` | 1 | 1 |
| `cpu_hpc_192c_representative` | `node001` | `node001:cpu` | 1 | 1 |
| `cpu_hpc_192c_equivalence` | `node003` | `node003:cpu` | 0 | 0 |
| `cpu_jtl110_representative` | `windows_a` | `windows_a:cpu` | 1 | 1 |

## Selected Rows

| Lane | Workload | Node | State | Missing profiles | Command |
|---|---|---|---|---|---|
| `gpu_jtl110_representative` | `gpu_cnn_torch_resnet50` | `gpu_a` | `full_loaded` | `5,6,8` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --tier T0_core_curve --resource-state full_loaded --nodes gpu_a --workloads gpu_cnn_torch_resnet50 --max-rows 1 --run-prefix ff_parallel_eta_t0_full_20260630_gpu_jtl110_representative_gpu_a_full_loaded_gpu_cnn_torch_resnet50 --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl110_representative_gpu_a_full_loaded_gpu_cnn_torch_resnet50.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl110_representative_gpu_a_full_loaded_gpu_cnn_torch_resnet50.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl110_representative_gpu_a_full_loaded_gpu_cnn_torch_resnet50.md` |
| `gpu_jtl311_cpu_fast` | `gpu_cnn_torch_resnet50` | `cpu_a` | `full_loaded` | `1,2,3,4,5,6,8` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --tier T0_core_curve --resource-state full_loaded --nodes cpu_a --workloads gpu_cnn_torch_resnet50 --max-rows 1 --run-prefix ff_parallel_eta_t0_full_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_gpu_cnn_torch_resnet50 --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_gpu_cnn_torch_resnet50.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_gpu_cnn_torch_resnet50.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_jtl311_cpu_fast_cpu_a_full_loaded_gpu_cnn_torch_resnet50.md` |
| `gpu_node007_direct` | `gpu_cnn_torch_resnet50` | `node007-direct` | `full_loaded` | `1,2,3,4,5,6,8` | `python3 -m algorithm.experiments.full_factorial_eta_probe_runner --allow-launch --tier T0_core_curve --resource-state full_loaded --nodes node007-direct --workloads gpu_cnn_torch_resnet50 --max-rows 1 --run-prefix ff_parallel_eta_t0_full_20260630_gpu_node007_direct_node007-direct_full_loaded_gpu_cnn_torch_resnet50 --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_node007_direct_node007-direct_full_loaded_gpu_cnn_torch_resnet50.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_node007_direct_node007-direct_full_loaded_gpu_cnn_torch_resnet50.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_gpu_node007_direct_node007-direct_full_loaded_gpu_cnn_torch_resnet50.md` |
| `cpu_hpc_192c_representative` | `light_control_local` | `node001` | `full_loaded` | `1,2,4,8,13,14` | `python3 -m algorithm.experiments.full_factorial_cpu_eta_probe_runner --allow-launch --tier T0_core_curve --resource-state full_loaded --nodes node001 --workloads light_control_local --max-rows 1 --run-prefix ff_parallel_eta_t0_full_20260630_cpu_hpc_192c_representative_node001_full_loaded_light_control_local --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_hpc_192c_representative_node001_full_loaded_light_control_local.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_hpc_192c_representative_node001_full_loaded_light_control_local.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_hpc_192c_representative_node001_full_loaded_light_control_local.md` |
| `cpu_jtl110_representative` | `light_control_local` | `windows_a` | `full_loaded` | `1,2,4,8,13,14` | `python3 -m algorithm.experiments.full_factorial_cpu_eta_probe_runner --allow-launch --tier T0_core_curve --resource-state full_loaded --nodes windows_a --workloads light_control_local --max-rows 1 --run-prefix ff_parallel_eta_t0_full_20260630_cpu_jtl110_representative_windows_a_full_loaded_light_control_local --output md/experiment_artifacts/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_jtl110_representative_windows_a_full_loaded_light_control_local.json --cache-output md/experiment_artifacts/service_cache_v2_full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_jtl110_representative_windows_a_full_loaded_light_control_local.json --markdown-output md/full_factorial_parallel_probe_ff_parallel_eta_t0_full_20260630_cpu_jtl110_representative_windows_a_full_loaded_light_control_local.md` |

## Claim Boundary

This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service cache only after their runner writes task-native tqdm/progress stable-rate summaries.
