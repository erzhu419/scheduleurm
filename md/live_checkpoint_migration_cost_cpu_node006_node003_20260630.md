# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint_node006_to_node003` | `pure_cpu` | 0.25 | `node006->node003` | `MEASURED` | 20.134 | 0.179 | 18.903 | 0.968 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node006_to_node003` | `pure_cpu` | 0.50 | `node006->node003` | `MEASURED` | 17.099 | 0.161 | 15.871 | 0.982 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node006_to_node003` | `pure_cpu` | 0.75 | `node006->node003` | `MEASURED` | 16.915 | 0.160 | 15.662 | 1.009 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
