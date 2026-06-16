# Corner-Case Benchmark Plan

JSON artifact: `md/experiment_artifacts/corner_case_plan_20260613/corner_case_benchmark_plan.json`

This plan is the experiment-side contract for the current empty-resource window. It uses benchmark logs with explicit `rate=` and `ETA` lines as the ETA source of truth.

## Resource Population

- GPU nodes: jtl110gpu, jtl110gpu2, node007-direct
- CPU nodes: node001, node002, node003, node004, node005, node006

## ETA Cache Contract

- `workload_key`
- `command_fingerprint`
- `normalized_parameters`
- `node`
- `device_family`
- `device_model`
- `device_count`
- `gpu_indices`
- `cpu_worker_count`
- `co_location_profile`
- `resource_occupancy_state`
- `algorithm_mode`
- `hard_rule_mode`
- `python_or_conda_env`
- `cuda_visible_devices`
- `driver_or_runtime_version`

## Scenarios

| id | quadrant | resource state | nodes | policies |
|---|---|---|---|---|
| `gpu_empty_single_eta` | `q01_gpu_heavy` | `empty` | jtl110gpu, jtl110gpu2, node007-direct | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `gpu_half_loaded_add_one` | `q01_gpu_heavy` | `half_loaded` | jtl110gpu, jtl110gpu2, node007-direct | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `gpu_full_loaded_add_one` | `q01_gpu_heavy` | `full_loaded` | jtl110gpu, jtl110gpu2, node007-direct | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `node007_30gb_multigpu_plus_small` | `q11_cpu_gpu_coupled` | `large_multigpu_resident` | node007-direct | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `cpu_empty_single_eta` | `q10_cpu_host_bound` | `empty` | node001, node002, node003, node004, node005, node006 | legacy, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, iadeep_interference_aware |
| `cpu_half_loaded_add_small` | `q10_cpu_host_bound` | `half_loaded` | node001, node002, node003, node004, node005, node006 | legacy, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, iadeep_interference_aware |
| `cpu_full_loaded_add_small` | `q10_cpu_host_bound` | `full_loaded` | node001, node002, node003, node004, node005, node006 | legacy, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, iadeep_interference_aware |
| `mixed_gpu_cpu_resident_plus_small` | `q11_cpu_gpu_coupled` | `mixed_loaded` | node007-direct, node001 | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `queue_backlog_eta_load_balance` | `portfolio` | `queued_backlog` | jtl110gpu, jtl110gpu2, node007-direct, node001, node002 | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |
| `eta_cache_identical_signature` | `portfolio` | `cache_reuse` | local | legacy, sweetspot_v1, theorem_maxweight_v1, gavel_finish_time_fairness, pollux_goodput_resize, sia_multi_cluster_goodput, gandiva_pack_time_slice, salus_memory_pack, iadeep_interference_aware |

## Claim Boundary

This plan can support OR claim-facing experiments only after the corresponding probe summaries show stable ETA windows and the replay/live gate consumes those measured rows.  It is not itself a performance claim.
