# Module69 FreqDuet promoted ep100 c65p Completed-History Certificate

```text
run_id = module69_freqduet_promoted_ep100_c65p_completed_history
sub_bucket = freqduet_cpu_ablation|c_65p
workload_key = freqduet_promoted_ep100_c65p_completed_history
record_count = 6
total_units_episode = 60000.000000
profile_domain = [1]
min_realized_episode_s = 1.727478545463
median_realized_episode_s = 1.895149322407
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU c65p run_freqduet_promoted_ep100_hpc_batch.sh records with explicit job-start/job-end bounds and the verified EPISODES=100 script default. It does not map other shell batch scripts or non-ep100 overrides.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed records used for service | 6 |
| Parsed episode units | 60000 |
| Minimum realized episode/s | 1.727478545463 |
| P10 realized episode/s | 1.727478545463 |
| Median realized episode/s | 1.895149322407 |
| Maximum duration s | 6225.216018 |

## Artifacts

```text
md/experiment_artifacts/module69_freqduet_promoted_ep100_c65p_completed_history.json
md/experiment_artifacts/module69_freqduet_promoted_ep100_c65p_completed_history.md
md/experiment_artifacts/module69_freqduet_promoted_ep100_c65p_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | episode/s |
|---|---|---:|---:|---:|---:|
| `t6099` | `FreqDuet` | 95 | 9500 | 5499.344710 | 1.727478545463 |
| `t6097` | `FreqDuet` | 116 | 11600 | 6225.216018 | 1.863389152597 |
| `t6096` | `FreqDuet` | 84 | 8400 | 4496.488076 | 1.868124602596 |
| `t6098` | `FreqDuet` | 95 | 9500 | 4942.320410 | 1.922174042219 |
| `t6100` | `FreqDuet` | 94 | 9400 | 2652.838878 | 3.543373884896 |
| `t6101` | `FreqDuet` | 116 | 11600 | 2753.970303 | 4.212100612748 |

## Interpretation

Strict completed-history lower-service certificate for c65p promoted ep100 shell-batch records. The local script was audited: each combined job is forwarded to an underlying runner with EPISODES=100, so work units are job-count times 100 episodes. Only profile 1 is loaded.
