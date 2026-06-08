# Module63 Transit Native Promotion c17_32 Seed-Range Completed-History Certificate

```text
run_id = module63_transit_native_promotion_c17_32_seedrange_completed_history
scope = native_promotion_replan_validation, no GPU, 17-32 CPU cores, explicit seed-index range
workload_key = transit_native_promotion_c17_32_seedrange_completed_history
record_count = 71
total_units_seed_episode = 17760.000000
profile_domain = [1]
min_realized_seed_episode_s = 0.075456066396
median_realized_seed_episode_s = 0.146156842791
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `native_promotion_replan_validation` production records with 17-32 requested CPU cores and explicit `--seed-index-start/--seed-index-end` plus `--episodes` work units. It does not map direct runner_v3, paper longtrain shell wrappers, or native commands without seed-index ranges.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 71 |
| Parsed seed-episode units | 17760 |
| Minimum realized seed-episode/s | 0.075456066396 |
| P10 realized seed-episode/s | 0.094578986838 |
| Median realized seed-episode/s | 0.146156842791 |
| Maximum duration s | 13570.810790 |

## Artifacts

```text
md/experiment_artifacts/module63_transit_native_promotion_c17_32_seedrange_completed_history.json
md/experiment_artifacts/module63_transit_native_promotion_c17_32_seedrange_completed_history.md
md/experiment_artifacts/module63_transit_native_promotion_c17_32_seedrange_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Seed-episode/s |
|---|---|---:|---:|---:|---:|
| `t7273` | `TransitDuet` | 32 | 1024 | 13570.810790 | 0.075456066396 |
| `t7274` | `TransitDuet` | 32 | 1024 | 13567.081509 | 0.075476807546 |
| `t7276` | `TransitDuet` | 32 | 1024 | 13564.088950 | 0.075493459514 |
| `t7275` | `TransitDuet` | 32 | 1024 | 13563.821040 | 0.075494950648 |
| `t7277` | `TransitDuet` | 32 | 1024 | 12464.244648 | 0.082154998474 |
| `t7278` | `TransitDuet` | 32 | 1024 | 12460.628850 | 0.082178838030 |
| `t7969` | `TransitDuet` | 32 | 48 | 514.563776 | 0.093282897505 |
| `t7963` | `TransitDuet` | 32 | 48 | 507.512309 | 0.094578986838 |

## Interpretation

Strict completed-history lower-service certificate for c_17_32 native_promotion_replan_validation production records with explicit seed-index ranges. Only profile 1 is loaded; native commands without seed ranges, direct runners, paper longtrain shell wrappers, and higher co-location profiles are not claimed.
