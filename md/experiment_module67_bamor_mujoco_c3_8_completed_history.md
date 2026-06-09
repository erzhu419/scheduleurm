# Module67 BAMOR c3_8 Mujoco Completed-History Certificate

```text
run_id = module67_bamor_mujoco_c3_8_completed_history
script = train_bamor_mujoco.py
workload_key = bamor_mujoco_c3_8_completed_history
record_count = 76
total_units_training_step = 3800000.000000
profile_domain = [1]
min_realized_training_step_s = 37.510192457006
median_realized_training_step_s = 75.308471170661
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `train_bamor_mujoco.py` BAMOR production records requesting 3-8 CPU cores with parseable training-step work units. It does not map other BAMOR CPU buckets, unparseable commands, or higher co-location profiles.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 76 |
| Parsed training steps | 3800000 |
| Minimum realized training-step/s | 37.510192457006 |
| P10 realized training-step/s | 44.544769837481 |
| Median realized training-step/s | 75.308471170661 |
| Maximum duration s | 1332.971033 |

## Artifacts

```text
md/experiment_artifacts/module67_bamor_mujoco_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_mujoco_c3_8_completed_history.md
md/experiment_artifacts/module67_bamor_mujoco_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | CPU | Units | Duration s | Training-step/s |
|---|---:|---:|---:|---:|
| `t8052` | 8 | 50000 | 1332.971033 | 37.510192457006 |
| `t8050` | 8 | 50000 | 1257.063712 | 39.775231365315 |
| `t7985` | 8 | 50000 | 1238.935756 | 40.357217691689 |
| `t8051` | 8 | 50000 | 1228.818109 | 40.689504534278 |
| `t8449` | 8 | 50000 | 1189.733876 | 42.026205212016 |
| `t8453` | 8 | 50000 | 1166.695406 | 42.856087136487 |
| `t8455` | 8 | 50000 | 1158.337454 | 43.165314066173 |
| `t8447` | 8 | 50000 | 1122.466233 | 44.544769837481 |

## Interpretation

Strict completed-history lower-service certificate for train_bamor_mujoco.py BAMOR c_3_8 production records with parseable training-step units. Only profile 1 is loaded. This script-level split supersedes the broad Module64 BAMOR c_3_8 class for production-capacity certification and does not claim BAMOR c_le2, c9_16, c17_32, or unparseable command shapes.
