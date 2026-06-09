# Module68 Transit Native Promotion c33_64 Batch Completed-History Certificate

```text
run_id = module68_transit_native_promotion_c33_64_batch_completed_history
sub_bucket = freqduet_cpu_ablation|c_33_64
workload_key = transit_native_promotion_c33_64_batch_completed_history
record_count = 45
total_units_seed_episode = 5922.000000
profile_domain = [1]
min_realized_seed_episode_s = 0.060204321136
median_realized_seed_episode_s = 0.154302751178
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `c_33_64 native_promotion_replan_validation batch records with parsed seed-count times episodes`. It does not map direct `runner_v3.py`, c_65p/c_17_32 native commands, or native commands whose seed work units cannot be parsed without executing code.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 45 |
| Parsed seed-episode units | 5922 |
| Minimum realized seed-episode/s | 0.060204321136 |
| P10 realized seed-episode/s | 0.092566768555 |
| Median realized seed-episode/s | 0.154302751178 |
| Maximum duration s | 2014.511699 |

## Artifacts

```text
md/experiment_artifacts/module68_transit_native_promotion_c33_64_batch_completed_history.json
md/experiment_artifacts/module68_transit_native_promotion_c33_64_batch_completed_history.md
md/experiment_artifacts/module68_transit_native_promotion_c33_64_batch_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Seed-episode/s |
|---|---|---:|---:|---:|---:|
| `t6075` | `TransitDuet` | 60 | 32 | 531.523309 | 0.060204321136 |
| `t5952` | `transit_hrl` | 60 | 32 | 394.805516 | 0.081052565686 |
| `t6250` | `FreqHRL` | 43 | 86 | 1026.328111 | 0.083793865825 |
| `t6252` | `FreqHRL` | 43 | 85 | 986.852039 | 0.086132466278 |
| `t6247` | `FreqHRL` | 43 | 86 | 929.059114 | 0.092566768555 |
| `t6369` | `FreqHRL` | 43 | 171 | 1789.410113 | 0.095562218393 |
| `t6249` | `FreqHRL` | 43 | 85 | 873.536416 | 0.097305617126 |
| `t6180` | `FreqHRL` | 43 | 86 | 871.527398 | 0.098677333824 |

## Interpretation

Strict completed-history lower-service certificate for c_33_64 native_promotion_replan_validation batch records with parsed seed-count times episodes. The parser accepts explicit CLI --seeds, --seed-index-start/--seed-index-end, and Python AST seed comprehensions inside direct or bash -lc wrapped python -c commands. Only profile 1 is loaded; runner_v3 and unparseable command shapes remain outside this class.
