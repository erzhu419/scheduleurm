# Module64 BAMOR c3_8 CPU Training Completed-History Certificate

```text
run_id = module64_bamor_cpu_training_c3_8_completed_history
scope = BAMOR no-GPU CPU training, 3-8 CPU cores, parseable training-step units
workload_key = bamor_cpu_training_c3_8_completed_history
record_count = 142
total_units_training_step = 25252000.000000
profile_domain = [1]
min_realized_training_step_s = 10.055276060265
median_realized_training_step_s = 215.355053987056
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU BAMOR production records requesting 3-8 CPU cores when `train_compare_baselines.py`, `train_bamor_mujoco.py`, or `run_bamor_diagnostic_shard.py` exposes explicit training-step work units. It does not map BAMOR c_le2/c9_16/c17_32 records, `--method all`, or commands without parseable step counts.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-history records | 142 |
| Parsed training-step units | 25252000 |
| Minimum realized training-step/s | 10.055276060265 |
| P10 realized training-step/s | 61.679100127895 |
| Median realized training-step/s | 215.355053987056 |
| Maximum duration s | 1332.971033 |

## Script Breakdown

| Script | Count |
|---|---:|
| `run_bamor_diagnostic_shard.py` | 25 |
| `train_bamor_mujoco.py` | 59 |
| `train_compare_baselines.py` | 58 |

## Artifacts

```text
md/experiment_artifacts/module64_bamor_cpu_training_c3_8_completed_history.json
md/experiment_artifacts/module64_bamor_cpu_training_c3_8_completed_history.md
md/experiment_artifacts/module64_bamor_cpu_training_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Script | CPU | Units | Duration s | Training-step/s |
|---|---|---:|---:|---:|---:|
| `t7538` | `train_compare_baselines.py` | 8 | 1000 | 99.450278 | 10.055276060265 |
| `t7539` | `train_compare_baselines.py` | 8 | 1000 | 97.164064 | 10.291870903722 |
| `t8052` | `train_bamor_mujoco.py` | 8 | 50000 | 1332.971033 | 37.510192457006 |
| `t8050` | `train_bamor_mujoco.py` | 8 | 50000 | 1257.063712 | 39.775231365315 |
| `t7985` | `train_bamor_mujoco.py` | 8 | 50000 | 1238.935756 | 40.357217691689 |
| `t8051` | `train_bamor_mujoco.py` | 8 | 50000 | 1228.818109 | 40.689504534278 |
| `t7993` | `train_bamor_mujoco.py` | 8 | 50000 | 1098.789016 | 45.504641255687 |
| `t8128` | `train_bamor_mujoco.py` | 8 | 50000 | 1090.333813 | 45.857515749811 |

## Interpretation

Strict completed-history lower-service certificate for BAMOR c_3_8 CPU training records. The parser accepts train_compare_baselines.py, train_bamor_mujoco.py, and run_bamor_diagnostic_shard.py only when training-step units are explicit. Only profile 1 is loaded; BAMOR c_le2/c9_16/c17_32 buckets and unparseable commands are not claimed.
