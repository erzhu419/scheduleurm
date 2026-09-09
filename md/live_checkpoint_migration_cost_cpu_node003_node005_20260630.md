# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.25 | `node003->node005` | `MEASURED` | 16.689 | 0.165 | 15.472 | 0.967 | 0.085 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.50 | `node003->node005` | `MEASURED` | 16.655 | 0.162 | 15.389 | 1.021 | 0.083 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.75 | `node003->node005` | `MEASURED` | 16.631 | 0.164 | 15.405 | 0.979 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
