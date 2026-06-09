# Module72 CFCMT Snapshot Generation c_le2 Completed-History Certificate

```text
run_id = module72_cfcmt_snapshot_generation_c_le2_completed_history
workload_key = cfcmt_snapshot_generation_c_le2_completed_history
record_count = 5
mapped_completed_active_count = 5
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_snapshot_window_s = 0.028279002327
```

## Unit Rule

Strict completed-history lower-service certificate for CFCMT SUMO/APC/AVL snapshot-generation records. It maps only no-GPU c_le2 snapshot_generation commands with snapshot-window units inferred from --snapshot-period and the stage2-report duration family.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 5 |
| done records used | 5 |
| total parsed units | 1560.000000 |
| min realized rate | 0.028279002327 |
| median realized rate | 0.133834433748 |

## Interpretation

Only profile 1 is loaded. Snapshot-window units are separate from simulated-second generation and policy-event rollout units.

## Artifact Paths

```text
md/experiment_artifacts/module72_cfcmt_snapshot_generation_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_snapshot_generation_c_le2_completed_history.md
md/experiment_artifacts/module72_cfcmt_snapshot_generation_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
