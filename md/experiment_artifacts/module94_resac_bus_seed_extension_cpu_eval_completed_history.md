# module94_resac_bus_seed_extension_cpu_eval_completed_history

Date: 2026-06-11

Strict completed-history lower-service certificate for residual RE-SAC bus seed-extension CPU eval commands. The progress unit is one completed command.

```text
workload_key = resac_bus_seed_extension_cpu_eval_completed_history
record_count = 2
completed_record_count = 2
profile_domain = [1]
lower_service = 0.000009542466 command/s
total_units = 2.000000 command
completed_units = 2.000000 command
theorem_status = strict_completed_history_lower_service
```

## Scope

This module uses command-level completed-history service. Done records with realized wall-clock duration define the profile-1 lower-service point; running records, when present, contribute arrival load but not service samples. Cancelled, failed, and forgotten records are not mapped by Module94 helpers.

## Artifacts

```text
md/experiment_artifacts/module94_resac_bus_seed_extension_cpu_eval_completed_history.json
md/experiment_artifacts/module94_resac_bus_seed_extension_cpu_eval_completed_history.md
md/experiment_artifacts/module94_resac_bus_seed_extension_cpu_eval_completed_history_reports/profile_1_per_resource_summary.json
```
