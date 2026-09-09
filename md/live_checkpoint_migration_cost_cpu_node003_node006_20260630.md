# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint_node003_to_node006` | `pure_cpu` | 0.25 | `node003->node006` | `MEASURED` | 16.758 | 0.161 | 15.523 | 0.991 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node003_to_node006` | `pure_cpu` | 0.50 | `node003->node006` | `MEASURED` | 16.665 | 0.161 | 15.446 | 0.974 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node003_to_node006` | `pure_cpu` | 0.75 | `node003->node006` | `MEASURED` | 16.960 | 0.168 | 15.725 | 0.983 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
