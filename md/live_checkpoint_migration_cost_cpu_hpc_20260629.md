# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `3`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.25 | `node003->node005` | `MEASURED` | 16.359 | 0.174 | 15.149 | 0.952 | 0.083 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.50 | `node003->node005` | `MEASURED` | 16.152 | 0.170 | 14.948 | 0.950 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.75 | `node003->node005` | `MEASURED` | 16.224 | 0.170 | 15.041 | 0.930 | 0.083 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
