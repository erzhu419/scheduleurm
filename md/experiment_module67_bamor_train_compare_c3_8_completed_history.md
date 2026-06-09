# Module67 BAMOR c3_8 Train-Compare Completed-History Certificate

```text
run_id = module67_bamor_train_compare_c3_8_completed_history
script = train_compare_baselines.py
workload_key = bamor_train_compare_c3_8_completed_history
record_count = 58
total_units_training_step = 5602000.000000
profile_domain = [1]
min_realized_training_step_s = 10.055276060265
median_realized_training_step_s = 224.855712575829
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `train_compare_baselines.py` BAMOR production records requesting 3-8 CPU cores with parseable training-step work units. It does not map other BAMOR CPU buckets, unparseable commands, or higher co-location profiles.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 58 |
| Parsed training steps | 5602000 |
| Minimum realized training-step/s | 10.055276060265 |
| P10 realized training-step/s | 209.163961556647 |
| Median realized training-step/s | 224.855712575829 |
| Maximum duration s | 484.693430 |

## Artifacts

```text
md/experiment_artifacts/module67_bamor_train_compare_c3_8_completed_history.json
md/experiment_artifacts/module67_bamor_train_compare_c3_8_completed_history.md
md/experiment_artifacts/module67_bamor_train_compare_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | CPU | Units | Duration s | Training-step/s |
|---|---:|---:|---:|---:|
| `t7538` | 8 | 1000 | 99.450278 | 10.055276060265 |
| `t7539` | 8 | 1000 | 97.164064 | 10.291870903722 |
| `t7563` | 8 | 100000 | 484.693430 | 206.315979900462 |
| `t7564` | 8 | 100000 | 481.384038 | 207.734349525568 |
| `t7565` | 8 | 100000 | 479.127442 | 208.712737485826 |
| `t7566` | 8 | 100000 | 478.093832 | 209.163961556647 |
| `t7567` | 8 | 100000 | 475.969359 | 210.097558016517 |
| `t7568` | 8 | 100000 | 473.661282 | 211.121330447131 |

## Interpretation

Strict completed-history lower-service certificate for train_compare_baselines.py BAMOR c_3_8 production records with parseable training-step units. Only profile 1 is loaded. This script-level split supersedes the broad Module64 BAMOR c_3_8 class for production-capacity certification and does not claim BAMOR c_le2, c9_16, c17_32, or unparseable command shapes.
