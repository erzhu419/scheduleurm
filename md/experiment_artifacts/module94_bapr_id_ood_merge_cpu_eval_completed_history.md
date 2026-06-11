# module94_bapr_id_ood_merge_cpu_eval_completed_history

Date: 2026-06-11

Strict completed-history lower-service certificate for residual BAPR id/ood merge CPU eval commands. The progress unit is one completed command.

```text
workload_key = bapr_id_ood_merge_cpu_eval_completed_history
record_count = 3
completed_record_count = 3
profile_domain = [1]
lower_service = 0.004724246710 command/s
total_units = 3.000000 command
completed_units = 3.000000 command
theorem_status = strict_completed_history_lower_service
```

## Scope

This module uses command-level completed-history service. Done records with realized wall-clock duration define the profile-1 lower-service point; running records, when present, contribute arrival load but not service samples. Cancelled, failed, and forgotten records are not mapped by Module94 helpers.

## Artifacts

```text
md/experiment_artifacts/module94_bapr_id_ood_merge_cpu_eval_completed_history.json
md/experiment_artifacts/module94_bapr_id_ood_merge_cpu_eval_completed_history.md
md/experiment_artifacts/module94_bapr_id_ood_merge_cpu_eval_completed_history_reports/profile_1_per_resource_summary.json
```
