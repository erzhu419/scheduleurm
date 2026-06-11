# module93_freqduet_spacectx_ep100_c3_8_completed_history

Date: 2026-06-11

Strict completed-history lower-service certificate for FreqDuet spacectx_screen_ep100_wu10 c3_8 production shards. The progress unit is one completed episode.

```text
workload_key = freqduet_spacectx_ep100_c3_8_completed_history
record_count = 5
completed_record_count = 3
profile_domain = [1]
lower_service = 0.545634685015 episode/s
total_units = 13300.000000 episode
completed_units = 8100.000000 episode
theorem_status = strict_completed_history_lower_service
```

## Scope

This module intentionally splits spacectx_screen_ep100_wu10 episode shards from the broader module60 c3_8 FreqDuet ablation bucket. The five production records contribute arrival load; only the three completed shards with realized wall-clock duration contribute service samples. The lower-service value is the minimum completed episode/s rate.

## Artifacts

```text
md/experiment_artifacts/module93_freqduet_spacectx_ep100_c3_8_completed_history.json
md/experiment_artifacts/module93_freqduet_spacectx_ep100_c3_8_completed_history.md
md/experiment_artifacts/module93_freqduet_spacectx_ep100_c3_8_completed_history_reports/profile_1_per_resource_summary.json
```
