# Module62 FreqDuet c_le2 Runner V3 Completed-History Certificate

```text
run_id = module62_freqduet_runner_v3_c_le2_completed_history
sub_bucket = freqduet_cpu_ablation|c_le2
workload_key = freqduet_runner_v3_c_le2_completed_history
record_count = 84
total_units_episode = 4661.000000
profile_domain = [1]
min_realized_episode_s = 0.010924044632
median_realized_episode_s = 0.045454422208
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `runner_v3.py` production records with at most 2 requested CPU cores and explicit `--episodes` work units. It does not map c_le2 ablation, baseline-rule, scheduler wait, native validation, or shell-loop records.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 84 |
| Parsed episode units | 4661 |
| Minimum realized episode/s | 0.010924044632 |
| P10 realized episode/s | 0.014003016694 |
| Median realized episode/s | 0.045454422208 |
| Maximum duration s | 3265.216062 |

## Artifacts

```text
md/experiment_artifacts/module62_freqduet_runner_v3_c_le2_completed_history.json
md/experiment_artifacts/module62_freqduet_runner_v3_c_le2_completed_history.md
md/experiment_artifacts/module62_freqduet_runner_v3_c_le2_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Episode/s |
|---|---|---:|---:|---:|---:|
| `t5056` | `freqduet` | 1 | 1 | 91.541186 | 0.010924044632 |
| `t7921` | `freqduet` | 2 | 1 | 77.626541 | 0.012882191907 |
| `t4994` | `freqduet` | 1 | 1 | 77.023461 | 0.012983057184 |
| `t4871` | `freqduet` | 2 | 1 | 73.992434 | 0.013514895403 |
| `t5011` | `freq_transitduet` | 1 | 1 | 73.957688 | 0.013521244769 |
| `t5010` | `freq_transitduet` | 1 | 1 | 72.872299 | 0.013722635591 |
| `t5005` | `freqduet` | 2 | 1 | 72.809309 | 0.013734507447 |
| `t4990` | `freq_transitduet` | 2 | 1 | 72.317720 | 0.013827869631 |

## Interpretation

Strict completed-history lower-service certificate for direct c_le2 runner_v3.py production records with explicit episode counts. Only profile 1 is loaded; ablation, baseline-rule, scheduler-wait, native validation, and higher co-location profiles are not claimed.
