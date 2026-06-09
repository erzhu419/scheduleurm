# Module69 Transit Native Promotion c65p Completed-History Certificate

```text
run_id = module69_transit_native_promotion_c65p_completed_history
sub_bucket = freqduet_cpu_ablation|c_65p
workload_key = transit_native_promotion_c65p_completed_history
record_count = 48
total_units_seed_episode = 7026.000000
profile_domain = [1]
min_realized_seed_episode_s = 0.182131380046
median_realized_seed_episode_s = 0.372605968018
theorem_status = strict_completed_history_lower_service
```

## Scope

This module maps only no-GPU c65p native_promotion_replan_validation records with statically parseable seed-count times episodes. The parser accepts explicit CLI seeds, seed-index ranges, Python AST seed comprehensions, and shell command-substitution Python snippets that can be proved without execution.

## Completed-History Service

| Quantity | Value |
|---|---:|
| Completed records used for service | 48 |
| Parsed seed-episode units | 7026 |
| Minimum realized seed-episode/s | 0.182131380046 |
| P10 realized seed-episode/s | 0.218141716657 |
| Median realized seed-episode/s | 0.372605968018 |
| Maximum duration s | 1539.427988 |

## Artifacts

```text
md/experiment_artifacts/module69_transit_native_promotion_c65p_completed_history.json
md/experiment_artifacts/module69_transit_native_promotion_c65p_completed_history.md
md/experiment_artifacts/module69_transit_native_promotion_c65p_completed_history_reports/profile_1_per_resource_summary.json
```

## Slowest Records

| Task | Project | CPU | Units | Duration s | seed-episode/s |
|---|---|---:|---:|---:|---:|
| `t6688` | `FreqHRLNative` | 71 | 71 | 389.828485 | 0.182131380046 |
| `t6872` | `FreqHRLNative` | 71 | 71 | 358.168613 | 0.198230658235 |
| `t6690` | `FreqHRLNative` | 81 | 81 | 379.989573 | 0.213163743837 |
| `t6691` | `FreqHRLNative` | 81 | 81 | 376.352569 | 0.215223720395 |
| `t6692` | `FreqHRLNative` | 81 | 81 | 371.318248 | 0.218141716657 |
| `t8803` | `TransitDuet` | 96 | 341 | 1539.427988 | 0.221510848605 |
| `t6874` | `FreqHRLNative` | 81 | 81 | 346.158519 | 0.233996841340 |
| `t6875` | `FreqHRLNative` | 81 | 81 | 342.566992 | 0.236450101220 |
| `t6694` | `FreqHRLNative` | 72 | 72 | 287.795526 | 0.250177620875 |
| `t6791` | `FreqHRLNative` | 71 | 71 | 279.322039 | 0.254186888662 |
| `t6689` | `FreqHRLNative` | 99 | 99 | 385.075387 | 0.257092515438 |
| `t6693` | `FreqHRLNative` | 99 | 99 | 366.519725 | 0.270108245817 |

## Interpretation

Strict completed-history lower-service certificate for c65p native_promotion_replan_validation records. Only profile 1 is loaded; commands with unparseable dynamic seed logic remain outside this class.
