# Module70 Transit Trading Policy c_le2 Completed-History Certificate

```text
run_id = module70_transit_trading_policy_c_le2_completed_history
workload_key = transit_trading_policy_c_le2_completed_history
record_count = 18
mapped_completed_active_count = 18
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_policy_market_step_s = 650.109325868530
```

## Unit Rule

Strict completed-history lower-service certificate for low-CPU TransitDuet trading policy_entry and ppo_actor_critic records. It maps only c_le2 no-GPU policy-training commands with parsed train/eval market-step units.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 18 |
| done records used | 18 |
| total parsed units | 4147200.000000 |
| min realized rate | 650.109325868530 |
| median realized rate | 913.673605187209 |

## Interpretation

Only profile 1 is loaded. This class is separated from trading sweep validation because optimizer iterations change the work-unit semantics.

## Artifact Paths

```text
md/experiment_artifacts/module70_transit_trading_policy_c_le2_completed_history.json
md/experiment_artifacts/module70_transit_trading_policy_c_le2_completed_history.md
md/experiment_artifacts/module70_transit_trading_policy_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
