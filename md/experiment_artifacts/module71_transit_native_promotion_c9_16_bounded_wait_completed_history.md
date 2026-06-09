# Module71 Transit Native Promotion c9_16 Bounded-Wait Completed-History Certificate

```text
run_id = module71_transit_native_promotion_c9_16_bounded_wait_completed_history
workload_key = transit_native_promotion_c9_16_bounded_wait_completed_history
record_count = 7
mapped_completed_active_count = 7
theorem_status = strict_completed_history_lower_service
profile_domain = [1]
min_rate_seed_episode_s = 0.003792221485
```

## Unit Rule

Strict completed-history lower-service certificate for c_9_16 Transit/FreqHRL bounded_wait_nofinal_v14 native_promotion_replan_validation records. It maps only no-GPU native validation commands with that exact stress profile and statically parsed seed-count times episode units.

## Completed-History Lower Service

| Quantity | Value |
|---|---:|
| mapped completed-active records | 7 |
| done records used | 7 |
| total parsed units | 2192.000000 |
| min realized rate | 0.003792221485 |
| median realized rate | 0.003810675332 |

## Interpretation

Only profile 1 is loaded. This slow stress-profile class is isolated from faster c9_16 native profiles so the capacity LP uses a matching lower-service bound.

## Artifact Paths

```text
md/experiment_artifacts/module71_transit_native_promotion_c9_16_bounded_wait_completed_history.json
md/experiment_artifacts/module71_transit_native_promotion_c9_16_bounded_wait_completed_history.md
md/experiment_artifacts/module71_transit_native_promotion_c9_16_bounded_wait_completed_history_reports/profile_1_per_resource_summary.json
```
