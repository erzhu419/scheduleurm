# Module68 Transit Native Promotion c33_64 Single-Seed Completed-History Certificate

```text
run_id = module68_transit_native_promotion_c33_64_single_seed_completed_history
sub_bucket = freqduet_cpu_ablation|c_33_64
workload_key = transit_native_promotion_c33_64_single_seed_completed_history
record_count = 2
total_units_seed_episode = 2.000000
profile_domain = [1]
min_realized_seed_episode_s = 0.003367969314
median_realized_seed_episode_s = 0.003395821409
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU `c_33_64 native_promotion_replan_validation single-seed smoke/fix records`. It does not map direct `runner_v3.py`, c_65p/c_17_32 native commands, or native commands whose seed work units cannot be parsed without executing code.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed-active records | 2 |
| Parsed seed-episode units | 2 |
| Minimum realized seed-episode/s | 0.003367969314 |
| P10 realized seed-episode/s | 0.003367969314 |
| Median realized seed-episode/s | 0.003395821409 |
| Maximum duration s | 296.914819 |

## Artifacts

```text
md/experiment_artifacts/module68_transit_native_promotion_c33_64_single_seed_completed_history.json
md/experiment_artifacts/module68_transit_native_promotion_c33_64_single_seed_completed_history.md
md/experiment_artifacts/module68_transit_native_promotion_c33_64_single_seed_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | Seed-episode/s |
|---|---|---:|---:|---:|---:|
| `t6834` | `FreqHRLNative` | 38 | 1 | 296.914819 | 0.003367969314 |
| `t6835` | `FreqHRLNative` | 37 | 1 | 292.083926 | 0.003423673504 |

## Interpretation

Strict completed-history lower-service certificate for c_33_64 native_promotion_replan_validation single-seed smoke/fix records. The parser accepts explicit CLI --seeds, --seed-index-start/--seed-index-end, and Python AST seed comprehensions inside direct or bash -lc wrapped python -c commands. Only profile 1 is loaded; runner_v3 and unparseable command shapes remain outside this class.
