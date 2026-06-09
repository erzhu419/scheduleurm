# Module72 CFCMT Policy Rollout c_le2 Completed-History Certificate

```text
run_id = module72_cfcmt_policy_rollout_c_le2_completed_history
workload_key = cfcmt_policy_rollout_c_le2_completed_history
record_count = 10
mapped_completed_active_count = 10
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_policy_event_budget_s = 10.848108919228
```

## Unit Rule

Strict completed-history lower-service certificate for CFCMT SUMO policy-rollout validation records. It maps only no-GPU c_le2 policy_rollout_validation commands with policy-event-budget units.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 10 |
| done records used | 10 |
| total parsed units | 140820.000000 |
| min realized rate | 10.848108919228 |
| median realized rate | 53.247061775997 |

## Interpretation

Only profile 1 is loaded. Policy-rollout work is separated from SUMO generation because rollout commands reference generation outputs by path.

## Artifact Paths

```text
md/experiment_artifacts/module72_cfcmt_policy_rollout_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_policy_rollout_c_le2_completed_history.md
md/experiment_artifacts/module72_cfcmt_policy_rollout_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
