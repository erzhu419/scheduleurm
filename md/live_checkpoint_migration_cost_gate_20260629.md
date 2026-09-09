# Live Checkpoint Migration Cost Gate

- Status: `LIVE_CHECKPOINT_MIGRATION_COST_READY`
- Pass: `true`
- Measured rows: `9`
- Blocked rows: `0`
- Pending rows: `0`

| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `cnn_gpu_checkpoint` | `pure_gpu` | 0.25 | `jtl110gpu->jtl311linux` | `MEASURED` | 41.715 | 0.240 | 41.060 | 0.067 | 0.348 | true |
| `cnn_gpu_checkpoint` | `pure_gpu` | 0.50 | `jtl110gpu->jtl311linux` | `MEASURED` | 41.659 | 0.242 | 41.001 | 0.067 | 0.349 | true |
| `cnn_gpu_checkpoint` | `pure_gpu` | 0.75 | `jtl110gpu->jtl311linux` | `MEASURED` | 41.833 | 0.232 | 41.184 | 0.068 | 0.349 | true |
| `resac_halfcheetah_checkpoint` | `hybrid_rl` | 0.25 | `jtl110gpu->jtl311linux` | `MEASURED` | 20.834 | 0.125 | 20.467 | 0.066 | 0.176 | false |
| `resac_halfcheetah_checkpoint` | `hybrid_rl` | 0.50 | `jtl110gpu->jtl311linux` | `MEASURED` | 20.869 | 0.116 | 20.511 | 0.066 | 0.176 | false |
| `resac_halfcheetah_checkpoint` | `hybrid_rl` | 0.75 | `jtl110gpu->jtl311linux` | `MEASURED` | 20.780 | 0.124 | 20.415 | 0.066 | 0.176 | false |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.25 | `node003->node005` | `MEASURED` | 22.125 | 0.161 | 21.003 | 0.877 | 0.084 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.50 | `node003->node005` | `MEASURED` | 37.257 | 0.161 | 36.090 | 0.923 | 0.083 | true |
| `cpu_freqduet_surrogate_checkpoint` | `pure_cpu` | 0.75 | `node003->node005` | `MEASURED` | 47.678 | 0.162 | 46.481 | 0.952 | 0.084 | true |

Physical checkpoint/sync/resume timings for controlled benchmark payloads only.  No production task is migrated.
