# Full-Factorial ETA Lookup-Table Design

- Status: `DESIGN_READY`
- ETA design rows: `306`
- Migration design rows: `96`
- ETA CSV: `md/experiment_artifacts/full_factorial_eta_design_rows_cpu_all_state_lcb_20260727.csv`
- Migration CSV: `md/experiment_artifacts/full_factorial_migration_design_rows_cpu_all_state_lcb_20260727.csv`

## ETA Row Status

| Status | Rows |
|---|---:|
| `boundary_plus_pending_probe` | 12 |
| `measured` | 66 |
| `measured_to_capacity_boundary` | 10 |
| `partial_pending_probe` | 11 |
| `pending_probe` | 179 |
| `probe_defer_by_design` | 28 |

## Natural-Completion ETA Status

| Status | Rows |
|---|---:|
| `eta_boundary_plus_pending_probe` | 12 |
| `eta_measured` | 2 |
| `eta_measured_to_capacity_boundary` | 2 |
| `eta_pending_probe` | 179 |
| `probe_defer_by_design` | 28 |
| `service_measured_eta_pending` | 83 |

## Migration Row Status

| Status | Rows |
|---|---:|
| `pending_or_measured_live_gate` | 96 |

## Hardware Types

| Group | Primary node | Equivalent nodes | Role |
|---|---|---|---|
| `gpu_3080ti_12gb_dual` | `jtl110gpu` | `jtl110gpu2` | Full representative GPU matrix; jtl110gpu2 is homogeneous equivalence. |
| `gpu_3080ti_12gb_dual_equiv` | `jtl110gpu2` | `` | Equivalence validation for jtl110gpu; not a separate full matrix unless tolerance fails. |
| `gpu_rtx2080_8gb_dual_cpu_fast` | `jtl311linux` | `` | CPU-fast/GPU-weaker node; full GPU, RL, LLM-feasible, and host-contention matrix. |
| `gpu_node007_4x11gb` | `node007` | `` | Four homogeneous RTX 2080 Ti GPUs; bucket name is historical, live memory is about 11GB. |
| `cpu_hpc_192c` | `node001` | `node002, node003, node004, node005, node006` | Full CPU matrix on one 192-core node; equivalence rows on additional nodes. |
| `cpu_hpc_192c_equiv` | `node003` | `node005` | CPU equivalence/replication node for 192-core class. |
| `cpu_jtl110_128c` | `jtl110cpu` | `jtl110cpu2` | Independent CPU-server class; required for cross-CPU-family extrapolation. |

## Workload Families

| Quadrant | Workload | Env | Profile axis | Profiles | Runner |
|---|---|---|---|---:|---|
| `q00_low_cpu_low_gpu` | `light_control_local` | `light` | `colocation_count` | `[1, 2, 4, 8, 13, 14]` | `light_progress_control` |
| `q01_low_cpu_high_gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `colocation_count` | `[1, 2, 3, 4, 5, 6, 8]` | `torch_cnn_resnet50.cmd.tpl` |
| `q01_low_cpu_high_gpu` | `gpu_llm_distilgpt2` | `llm` | `colocation_count` | `[1, 2, 3, 4]` | `torch_llm_distilgpt2.cmd.tpl` |
| `q01_low_cpu_high_gpu` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `colocation_count` | `[1, 2, 3, 4, 5, 6, 8]` | `jax_matmul_progress` |
| `q10_high_cpu_low_gpu` | `cpu_heavy_local_bench` | `cpu` | `allocation_workers` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | `cpu_parallel_progress_benchmark` |
| `q10_high_cpu_low_gpu` | `freqduet_cpu_ablation_c17_32` | `freqduet` | `colocation_count` | `[1, 2, 4, 8, 16, 32, 64, 96, 128, 180, 192]` | `freqduet_or_checkpointable_surrogate` |
| `q10_high_cpu_low_gpu` | `sumo_eval_cpu` | `sumo` | `colocation_count` | `[1, 2, 4, 8, 16, 32, 64, 96, 128]` | `sumo_progress_or_surrogate` |
| `q11_high_cpu_high_gpu` | `hybrid_rl_resac_ant` | `ant` | `colocation_count` | `[1, 2, 3, 4, 5, 6]` | `resac_env_real_venv.cmd.tpl` |
| `q11_high_cpu_high_gpu` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `colocation_count` | `[1, 2, 3, 4, 5, 6]` | `resac_env_real_venv.cmd.tpl` |
| `q11_high_cpu_high_gpu` | `hybrid_rl_resac_hopper` | `hopper` | `colocation_count` | `[1, 2, 3, 4, 5, 6]` | `resac_env_real_venv.cmd.tpl` |
| `q11_high_cpu_high_gpu` | `hybrid_rl_resac_walker2d` | `walker2d` | `colocation_count` | `[1, 2, 3, 4, 5, 6]` | `resac_env_real_venv.cmd.tpl` |
| `q11_high_cpu_high_gpu` | `hybrid_rl_bapr_ant` | `bapr_ant` | `colocation_count` | `[1, 2, 3, 4, 5, 6]` | `bapr_ant_real.cmd.tpl` |

## Resident Load States

| State | Resident mix | Profile mode | Priority |
|---|---|---|---|
| `empty` | `none` | `all_profiles` | `required` |
| `half_loaded` | `same_workload_half_capacity` | `middle_profiles` | `required` |
| `full_loaded` | `same_workload_to_capacity_boundary` | `to_boundary` | `required` |
| `half_loaded` | `controlled_resident_workers=96` | `cpu_controlled_half` | `required` |
| `full_loaded` | `controlled_resident_workers=180` | `cpu_controlled_full` | `required` |
| `high_vram_resident` | `llm_or_memory_resident` | `add_one` | `required` |
| `cpu_resident` | `cpu_worker_resident` | `add_one` | `required` |
| `mixed_colocation` | `cnn_plus_llm` | `add_one` | `required` |
| `mixed_colocation` | `cnn_plus_hybrid_rl` | `add_one` | `required` |
| `mixed_colocation` | `llm_plus_hybrid_rl` | `add_one` | `required` |
| `mixed_colocation` | `cnn_plus_llm_plus_hybrid_rl` | `add_one` | `required` |
| `mixed_colocation` | `cpu_plus_gpu_target` | `add_one` | `required` |
| `unknown_env` | `description_only_or_unknown` | `probe_defer` | `admission` |

## Probe Order

1. T0 core curves: empty and same-workload profile ladders to stable row or capacity boundary.
2. T1 equivalence rows: homogeneous-node validation, not a substitute for heterogeneous nodes.
3. T2 load-state rows: high-VRAM, CPU-resident, pairwise mixed, and triple mixed co-location.
4. Migration rows: controlled checkpoint/resume at 25%, 50%, and 75% progress.
5. Rerun SOTA/ours replay only after all required rows for the declared claim are measured.

## Claim Boundary

This is the declared finite ETA lookup-table domain.  Rows marked measured are already in the service-cache snapshot.  Rows marked pending_probe must be measured with task-native tqdm/progress before they can enter theorem-facing replay.  This design is exhaustive over the listed hardware, workload, load-state, profile, and migration factors; it is not a claim about arbitrary future workloads.
