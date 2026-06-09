# Module72 CFCMT Feed Conversion c_le2 Completed-History Certificate

```text
run_id = module72_cfcmt_feed_conversion_c_le2_completed_history
workload_key = cfcmt_feed_conversion_c_le2_completed_history
record_count = 5
mapped_completed_active_count = 5
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_feed_conversion_s = 0.000676721109
```

## Unit Rule

Strict completed-history lower-service certificate for CFCMT GTFS/LTA feed conversion records. It maps only no-GPU c_le2 gtfs_to_h2o_xlsx.py and lta_to_h2o_xlsx.py commands.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 5 |
| done records used | 5 |
| total parsed units | 5.000000 |
| min realized rate | 0.000676721109 |
| median realized rate | 0.001601659547 |

## Interpretation

Only profile 1 is loaded. Feed conversion is separated from validation, simulation, snapshot, rollout, and traffic-signal work.

## Artifact Paths

```text
md/experiment_artifacts/module72_cfcmt_feed_conversion_c_le2_completed_history.json
md/experiment_artifacts/module72_cfcmt_feed_conversion_c_le2_completed_history.md
md/experiment_artifacts/module72_cfcmt_feed_conversion_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```
