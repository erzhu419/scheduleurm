# Module72 CFCMT SUMO Generation c_le2 Completed-History Certificate

```text
run_id = module72_cfcmt_sumo_generation_c_le2_completed_history
workload_key = cfcmt_sumo_generation_c_le2_completed_history
record_count = 4
mapped_completed_active_count = 4
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_sim_second_s = 18.527090540978
```

## Unit Rule

Strict completed-history lower-service certificate for CFCMT SUMO/APC/AVL generation records. It maps only no-GPU c_le2 sumo_apc_avl_sumo_generation commands with explicit --duration-sec simulated-second units.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 4 |
| done records used | 4 |
| total parsed units | 106200.000000 |
| min realized rate | 18.527090540978 |
| median realized rate | 136.069674746081 |

## Interpretation

Only profile 1 is loaded. Policy rollout and snapshot-generation commands are matched before this class so stage2-report filenames cannot pull them into SUMO generation.

## Artifact Paths

```text
md/experiment_artifacts/module72_cfcmt_sumo_generation_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_sumo_generation_c_le2_completed_history.md
md/experiment_artifacts/module72_cfcmt_sumo_generation_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
