# Module67 BAMOR c3_8 Diagnostic-Shard Completed-History Certificate

```text
run_id = module67_bamor_diagnostic_shard_c3_8_completed_history
script = run_bamor_diagnostic_shard.py
workload_key = bamor_diagnostic_shard_c3_8_completed_history
record_count = 25
total_units_training_step = 16700000.000000
profile_domain = [1]
min_realized_training_step_s = 412.612802379794
median_realized_training_step_s = 1152.445308640983
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `run_bamor_diagnostic_shard.py` BAMOR production records requesting 3-8 CPU cores with parseable training-step work units. It does not map other BAMOR CPU buckets, unparseable commands, or higher co-location profiles.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 25 |
| Parsed training steps | 16700000 |
| Minimum realized training-step/s | 412.612802379794 |
| P10 realized training-step/s | 602.894428722710 |
| Median realized training-step/s | 1152.445308640983 |
| Maximum duration s | 1090.075526 |

## Artifacts

```text
md/experiment_artifacts/module67_bamor_diagnostic_shard_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_diagnostic_shard_c3_8_completed_history.md
md/experiment_artifacts/module67_bamor_diagnostic_shard_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | CPU | Units | Duration s | Training-step/s |
|---|---:|---:|---:|---:|
| `t7823` | 4 | 400000 | 969.431869 | 412.612802379794 |
| `t7882` | 5 | 500000 | 1090.075526 | 458.683813970620 |
| `t7879` | 5 | 500000 | 829.332593 | 602.894428722710 |
| `t7880` | 8 | 800000 | 980.189274 | 816.168898355755 |
| `t7806` | 4 | 400000 | 485.653547 | 823.632406752767 |
| `t7794` | 4 | 400000 | 482.553378 | 828.923841281651 |
| `t7871` | 8 | 800000 | 878.106319 | 911.051409988723 |
| `t7636` | 4 | 400000 | 410.942123 | 973.373079668407 |

## Interpretation

Strict completed-history lower-service certificate for run_bamor_diagnostic_shard.py BAMOR c_3_8 production records with parseable training-step units. Only profile 1 is loaded. This script-level split supersedes the broad Module64 BAMOR c_3_8 class for production-capacity certification and does not claim BAMOR c_le2, c9_16, c17_32, or unparseable command shapes.
