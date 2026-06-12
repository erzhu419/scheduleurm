# Production Launch / Completion Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `launch_status` | `WAIT_RESOURCE_OR_QUEUE` |
| `launch_safe` | false |
| `active_production_count` | 27 |
| `queued_production_count` | 0 |
| `running_production_count` | 27 |
| `local_gpu_max_util_pct` | 99.0 |
| `shadow_theorem_slot_count` | 4 |
| `shadow_trace_closed` | true |
| `progress_observation_count` | 24 |
| `large_scale_launched_completion_ready` | false |
| `large_scale_active_progress_ready` | true |

## Project Counts

| Project | Count |
|---|---:|
| `BAPR` | 24 |
| `FreqDuet` | 3 |

## Scope

Safe production closure gate.  It does not launch new work unless queued-production and low-utilization resource conditions are satisfied. In the current snapshot, it reports large-scale active-production progress plus non-invasive theorem shadow trace, not launched production completion.
