# Module70 Transit Native Control c_le2 Completed-History Certificate

```text
run_id = module70_transit_native_control_c_le2_completed_history
workload_key = transit_native_control_c_le2_completed_history
record_count = 7
mapped_completed_active_count = 7
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_native_control_episode_s = 0.028357295260
```

## Unit Rule

Strict completed-history lower-service certificate for low-CPU native_wait_credit_validation and native_real_demand_control_validation records with parsed native control episode units.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 7 |
| done records used | 7 |
| total parsed units | 148.000000 |
| min realized rate | 0.028357295260 |
| median realized rate | 0.070248930694 |

## Interpretation

Only profile 1 is loaded. This class is separated from native promotion-replan because the command semantics and realized lower-service differ.

## Artifact Paths

```text
md/experiment_artifacts/module70_transit_native_control_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_native_control_c_le2_completed_history.md
md/experiment_artifacts/module70_transit_native_control_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
