# module94_cfcmt_cpu_eval_completed_history

Date: 2026-06-11

Strict completed-history lower-service certificate for residual CFCMT cf_h2o eval production records. The progress unit is one completed command.

```text
workload_key = cfcmt_cpu_eval_completed_history
record_count = 71
completed_record_count = 71
profile_domain = [1]
lower_service = 0.000672555954 command/s
total_units = 71.000000 command
completed_units = 71.000000 command
theorem_status = strict_completed_history_lower_service
```

## Scope

This module uses command-level completed-history service. Done records with realized wall-clock duration define the profile-1 lower-service point; running records, when present, contribute arrival load but not service samples. Cancelled, failed, and forgotten records are not mapped by Module94 helpers.

## Artifacts

```text
md/experiment_artifacts/module94_cfcmt_cpu_eval_completed_history.json
md/experiment_artifacts/module94_cfcmt_cpu_eval_completed_history.md
md/experiment_artifacts/module94_cfcmt_cpu_eval_completed_history_reports/profile_1_per_resource_summary.json
```
