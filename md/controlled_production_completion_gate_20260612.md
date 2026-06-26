# Controlled Launched Completion Gate

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `CONTROLLED_32_AND_STRICT_HISTORY_COMPLETION_PASS` |
| `gate_pass` | true |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | true |
| `controlled_completion_strong_claim_ready` | true |
| `combined_controlled_and_history_completion_ready` | true |
| `pass_meaning` | 32-task controlled launched completion is closed.  The strict scheduler-history large-scale completion certificate is also closed as a separate evidence path; live oracle-traced large-scale completion remains false |
| `launch_status` | `LAUNCH_DISABLED` |
| `launch_safe` | false |
| `local_gpu_max_util_pct` | 0.0 |
| `bounded_controlled_completion_ready` | true |
| `controlled_6_task_completion_ready` | true |
| `controlled_32_task_completion_ready` | true |
| `organic_production_canary_recorder_ready` | true |
| `strict_history_large_scale_completion_ready` | true |
| `history_large_scale_completion_ready` | true |
| `live_oracle_traced_large_scale_completion_ready` | false |
| `live_trace_large_scale_completion_ready` | false |
| `production_wide_live_completion_ready` | false |
| `selected_controlled_run_path` | `<ARTIFACT_ROOT>/controlled_jtl110gpu_32_20260613.json` |
| `controlled_launched_task_count` | 32 |
| `controlled_completed_task_count` | 32 |
| `theorem_slot_count` | 32 |
| `candidate_count_total` | 56 |
| `candidate_count_per_launched_task` | 1.75 |
| `alpha0` | 0.0 |
| `alpha1` | 0.0 |
| `production_queued_count` | 0 |
| `production_progress_observation_count` | 4 |

## Scope

Recognizes existing bounded controlled launched completion evidence and defines the stronger 32-task controlled threshold.  Strict history completion and live-trace completion are separate fields. It only launches additional controlled tasks when explicitly allowed and GPU utilization is below the safety threshold.
