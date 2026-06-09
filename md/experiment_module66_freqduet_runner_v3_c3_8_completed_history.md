# Module66 FreqDuet c3_8 Runner V3 Completed-History Certificate

```text
run_id = module66_freqduet_runner_v3_c3_8_completed_history
sub_bucket = freqduet_cpu_ablation|c_3_8
workload_key = freqduet_runner_v3_c3_8_completed_history
record_count = 86
total_units_episode = 2709.000000
profile_domain = [1]
min_realized_episode_s = 0.006679260716
median_realized_episode_s = 0.043436045275
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU FreqDuet `runner_v3.py` production records requesting 3-8 CPU cores with explicit `--episodes` work units. It does not map c_3_8 `run_freqduet_ablation.py` records already covered by Module60, native validation commands, shell-expanded commands, or higher co-location profiles.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 86 |
| Parsed episode units | 2709 |
| Minimum realized episode/s | 0.006679260716 |
| P10 realized episode/s | 0.032252203573 |
| Median realized episode/s | 0.043436045275 |
| Maximum duration s | 2994.343364 |

## Artifacts

```text
md/experiment_artifacts/module66_freqduet_runner_v3_c3_8_completed_history.json
md/experiment_artifacts/module66_freqduet_runner_v3_c3_8_completed_history.md
md/experiment_artifacts/module66_freqduet_runner_v3_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Episode/s |
|---|---|---:|---:|---:|---:|
| `t4840` | `freqduet` | 3 | 20 | 2994.343364 | 0.006679260716 |
| `t4841` | `freqduet` | 3 | 20 | 2640.020293 | 0.007575699345 |
| `t4842` | `freqduet` | 4 | 20 | 1976.119231 | 0.010120846800 |
| `t4769` | `freqduet` | 3 | 1 | 73.249820 | 0.013651910770 |
| `t4876` | `freqduet` | 5 | 20 | 708.076977 | 0.028245516579 |
| `t4873` | `freqduet` | 5 | 20 | 663.839700 | 0.030127755219 |
| `t4864` | `freqduet` | 6 | 20 | 635.516427 | 0.031470468986 |
| `t5585` | `freqduet` | 4 | 40 | 1261.094563 | 0.031718477891 |

## Interpretation

Strict completed-history lower-service certificate for c_3_8 direct runner_v3.py production records with explicit episode counts. Only profile 1 is loaded; run_freqduet_ablation.py is handled by Module60, and native validation, shell-expanded, and higher co-location profiles are not claimed.
