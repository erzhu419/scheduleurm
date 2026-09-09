# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint_node005_to_node003` | `pure_cpu` | 0.25 | `node005->node003` | `MEASURED` | 17.051 | 0.164 | 15.757 | 1.045 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node005_to_node003` | `pure_cpu` | 0.50 | `node005->node003` | `MEASURED` | 16.765 | 0.168 | 15.529 | 0.985 | 0.083 | true |
| `cpu_freqduet_surrogate_checkpoint_node005_to_node003` | `pure_cpu` | 0.75 | `node005->node003` | `MEASURED` | 16.648 | 0.161 | 15.375 | 1.028 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
