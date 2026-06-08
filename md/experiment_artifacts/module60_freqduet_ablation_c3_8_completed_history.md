# Module60 FreqDuet c3_8 Ablation Completed-History Certificate

```text
run_id = module60_freqduet_ablation_c3_8_completed_history
sub_bucket = freqduet_cpu_ablation|c_3_8
workload_key = freqduet_cpu_ablation_c3_8_completed_history
record_count = 40
total_units_episode = 15173.000000
profile_domain = [1]
min_realized_episode_s = 0.011071600978
median_realized_episode_s = 0.198764896519
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `run_freqduet_ablation.py` production records with 3-8 requested CPU cores and parseable episode work units. It does not map c3_8 `runner_v3.py`, native validation scripts, or unknown-size shell loops.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 40 |
| Parsed episode units | 15173 |
| Minimum realized episode/s | 0.011071600978 |
| P10 realized episode/s | 0.111451562453 |
| Median realized episode/s | 0.198764896519 |
| Maximum duration s | 4288.845480 |

## Artifacts

```text
md/experiment_artifacts/module60_freqduet_ablation_c3_8_completed_history.json
md/experiment_artifacts/module60_freqduet_ablation_c3_8_completed_history.md
md/experiment_artifacts/module60_freqduet_ablation_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Episode/s |
|---|---|---:|---:|---:|---:|
| `t5238` | `FreqDuet` | 4 | 1 | 90.321174 | 0.011071600978 |
| `t6348` | `freqduet` | 3 | 2 | 95.240293 | 0.020999515452 |
| `t5015` | `freqduet` | 8 | 80 | 754.160687 | 0.106078189080 |
| `t5012` | `freqduet` | 8 | 80 | 717.800614 | 0.111451562453 |
| `t4845` | `freqduet` | 5 | 60 | 529.252756 | 0.113367383176 |
| `t4982` | `freqduet` | 5 | 200 | 1510.635383 | 0.132394621671 |
| `t4984` | `freqduet` | 5 | 200 | 1495.648028 | 0.133721300871 |
| `t7687` | `TransitDuet` | 5 | 200 | 1468.623702 | 0.136181923040 |

## Interpretation

Strict completed-history lower-service certificate for parseable c_3_8 run_freqduet_ablation.py production records. Only profile 1 is loaded; runner_v3 and higher co-location profiles are not claimed.
