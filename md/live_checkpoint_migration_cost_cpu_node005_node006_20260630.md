# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint_node005_to_node006` | `pure_cpu` | 0.25 | `node005->node006` | `MEASURED` | 16.904 | 0.191 | 15.586 | 1.028 | 0.099 | true |
| `cpu_freqduet_surrogate_checkpoint_node005_to_node006` | `pure_cpu` | 0.50 | `node005->node006` | `MEASURED` | 16.666 | 0.167 | 15.443 | 0.972 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node005_to_node006` | `pure_cpu` | 0.75 | `node005->node006` | `MEASURED` | 16.696 | 0.162 | 15.458 | 0.993 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
