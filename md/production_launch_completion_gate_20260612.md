# Production Launch / Completion Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `launch_status` | `WAIT_RESOURCE_OR_QUEUE` |
| `launch_safe` | false |
| `active_production_count` | 53 |
| `queued_production_count` | 4 |
| `running_production_count` | 49 |
| `local_gpu_max_util_pct` | 98.0 |
| `shadow_theorem_slot_count` | 8 |
| `shadow_trace_closed` | true |
| `progress_observation_count` | 20 |
| `large_scale_launched_completion_ready` | false |
| `large_scale_active_progress_ready` | true |

## Project Counts

| Project | Count |
|---|---:|
| `BAMOR` | 10 |
| `BAPR` | 20 |
| `CS-BAPR` | 11 |
| `FreqDuet` | 8 |

## Scope

Safe production closure gate.  It does not launch new work unless queued-production and low-utilization resource conditions are satisfied. In the current snapshot, it reports large-scale active-production progress plus non-invasive theorem shadow trace, not launched production completion.
