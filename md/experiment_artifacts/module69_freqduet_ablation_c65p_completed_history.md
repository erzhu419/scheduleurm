# Module69 FreqDuet c65p Ablation Completed-History Certificate

```text
run_id = module69_freqduet_ablation_c65p_completed_history
sub_bucket = freqduet_cpu_ablation|c_65p
workload_key = freqduet_cpu_ablation_c65p_completed_history
record_count = 7
total_units_episode = 18430.000000
profile_domain = [1]
min_realized_episode_s = 1.099735984153
median_realized_episode_s = 1.647391444607
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU c65p run_freqduet_ablation.py records with worker-threads fixed at one when specified and parsed jobs times episodes. It does not map promoted ep100 shell batches, native-promotion validation records, runner_v3 records, or unparseable command shapes.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed records used for service | 7 |
| Parsed episode units | 18430 |
| Minimum realized episode/s | 1.099735984153 |
| P10 realized episode/s | 1.099735984153 |
| Median realized episode/s | 1.647391444607 |
| Maximum duration s | 2909.789300 |

## Artifacts

```text
md/experiment_artifacts/module69_freqduet_ablation_c65p_completed_history.json
md/experiment_artifacts/module69_freqduet_ablation_c65p_completed_history.md
md/experiment_artifacts/module69_freqduet_ablation_c65p_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | episode/s |
|---|---|---:|---:|---:|---:|
| `t5645` | `FreqDuet` | 80 | 3200 | 2909.789300 | 1.099735984153 |
| `t5025` | `freq_transitduet` | 96 | 1210 | 748.040907 | 1.617558597169 |
| `t5647` | `FreqDuet` | 80 | 3200 | 1945.243494 | 1.645038273832 |
| `t5648` | `FreqDuet` | 80 | 3200 | 1942.464865 | 1.647391444607 |
| `t5643` | `FreqDuet` | 80 | 3200 | 1781.930308 | 1.795805361357 |
| `t5023` | `freq_transitduet` | 96 | 1220 | 662.119344 | 1.842568127729 |
| `t5646` | `FreqDuet` | 80 | 3200 | 1683.699360 | 1.900576834154 |

## Interpretation

Strict completed-history lower-service certificate for c65p run_freqduet_ablation.py production records. Only profile 1 is loaded; promoted ep100 shell batches and native-promotion validation are separate Module69 service classes.
