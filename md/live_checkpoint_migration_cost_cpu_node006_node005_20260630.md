# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint_node006_to_node005` | `pure_cpu` | 0.25 | `node006->node005` | `MEASURED` | 16.867 | 0.162 | 15.607 | 1.013 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node006_to_node005` | `pure_cpu` | 0.50 | `node006->node005` | `MEASURED` | 16.727 | 0.169 | 15.449 | 1.025 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint_node006_to_node005` | `pure_cpu` | 0.75 | `node006->node005` | `MEASURED` | 16.560 | 0.164 | 15.333 | 0.979 | 0.083 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
