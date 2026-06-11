# module94_scheduleurm_control_plane_completed_history

Date: 2026-06-11

Strict completed-history lower-service certificate for Scheduleurm control-plane production-certificate commands. The progress unit is one completed command.

```text
workload_key = scheduleurm_control_plane_completed_history
record_count = 55
completed_record_count = 55
profile_domain = [1]
lower_service = 0.000326719692 command/s
total_units = 55.000000 command
completed_units = 55.000000 command
theorem_status = strict_completed_history_lower_service
```

## Scope

This module uses command-level completed-history service. Done records with realized wall-clock duration define the profile-1 lower-service point; running records, when present, contribute arrival load but not service samples. Cancelled, failed, and forgotten records are not mapped by Module94 helpers.

## Artifacts

```text
md/experiment_artifacts/module94_scheduleurm_control_plane_completed_history.json
md/experiment_artifacts/module94_scheduleurm_control_plane_completed_history.md
md/experiment_artifacts/module94_scheduleurm_control_plane_completed_history_reports/profile_1_per_resource_summary.json
```
