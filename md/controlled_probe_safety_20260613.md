# Controlled Launched Completion Gate

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `BOUNDED_CONTROLLED_COMPLETION_PASS_32_TASK_FALSE_ORGANIC_FALSE` |
| `gate_pass` | true |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `pass_meaning` | bounded controlled launched completion and canary-recorder readiness, not 32-task controlled completion or organic production-wide completion |
| `launch_status` | `WAIT_RESOURCE` |
| `launch_safe` | false |
| `local_gpu_max_util_pct` | 0.0 |
| `bounded_controlled_completion_ready` | true |
| `controlled_32_task_completion_ready` | false |
| `organic_production_canary_recorder_ready` | true |
| `large_scale_organic_launched_completion_ready` | false |
| `controlled_launched_task_count` | 6 |
| `controlled_completed_task_count` | 6 |
| `theorem_slot_count` | 7 |
| `candidate_count_total` | 13 |
| `alpha0` | 0.0 |
| `alpha1` | 0.0 |
| `production_queued_count` | 0 |
| `production_progress_observation_count` | 0 |

## Scope

Recognizes existing bounded controlled launched completion evidence and defines the stronger 32-task controlled and organic production thresholds. It only launches additional controlled tasks when explicitly allowed and GPU utilization is below the safety threshold.
