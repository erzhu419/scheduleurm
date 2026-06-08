# Module61 FreqDuet c33_64 Ablation Completed-History Certificate

```text
run_id = module61_freqduet_ablation_c33_64_completed_history
sub_bucket = freqduet_cpu_ablation|c_33_64
workload_key = freqduet_cpu_ablation_c33_64_completed_history
record_count = 63
total_units_episode = 156140.000000
profile_domain = [1]
min_realized_episode_s = 0.391511590884
median_realized_episode_s = 1.260587867715
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `run_freqduet_ablation.py` production records with 33-64 requested CPU cores and parseable episode work units. It does not map c33_64 `runner_v3.py`, native validation scripts, or unknown-size shell loops.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 63 |
| Parsed episode units | 156140 |
| Minimum realized episode/s | 0.391511590884 |
| P10 realized episode/s | 0.997553747110 |
| Median realized episode/s | 1.260587867715 |
| Maximum duration s | 4291.060697 |

## Artifacts

```text
md/experiment_artifacts/module61_freqduet_ablation_c33_64_completed_history.json
md/experiment_artifacts/module61_freqduet_ablation_c33_64_completed_history.md
md/experiment_artifacts/module61_freqduet_ablation_c33_64_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Episode/s |
|---|---|---:|---:|---:|---:|
| `t7615` | `TransitDuet` | 42 | 1680 | 4291.060697 | 0.391511590884 |
| `t5149` | `freqduet` | 64 | 1520 | 2643.559357 | 0.574982360711 |
| `t5598` | `FreqDuet` | 48 | 2680 | 3347.458661 | 0.800607347646 |
| `t7614` | `TransitDuet` | 33 | 1320 | 1491.186367 | 0.885201226086 |
| `t7504` | `freqduet` | 36 | 1440 | 1592.636942 | 0.904160868122 |
| `t5627` | `FreqDuet` | 40 | 1600 | 1746.000534 | 0.916380017526 |
| `t5150` | `freqduet` | 64 | 1480 | 1483.629333 | 0.997553747110 |
| `t5565` | `FreqDuet` | 48 | 3200 | 3162.849486 | 1.011745900111 |

## Interpretation

Strict completed-history lower-service certificate for parseable c_33_64 run_freqduet_ablation.py production records. Only profile 1 is loaded; runner_v3, native validation, and higher co-location profiles are not claimed.
