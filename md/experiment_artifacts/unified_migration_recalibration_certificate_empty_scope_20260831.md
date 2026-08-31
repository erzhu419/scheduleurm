# Unified Migration Recalibration Certificate

- Status: `MIGRATION_RECALIBRATION_PASS`
- Pass: `true`
- Final cache SHA-256: `5eac04c5574e8b869271bc6e0ffce18ec0fc5627bfa016fb0a7c83cbc8865c57`
- Rows: `12`

| Action | Family | Progress | Old lower rate | New lower rate | Effective migration rate | Beneficial |
|---|---|---:|---:|---:|---:|---:|
| `migrate:cnn_jtl110gpu_to_jtl311linux:p25` | `pure_gpu` | 0.25 | 3.32354 | 2.60146 | 0.76255 | false |
| `migrate:cnn_jtl110gpu_to_jtl311linux:p50` | `pure_gpu` | 0.50 | 3.32354 | 2.60146 | 0.564001 | false |
| `migrate:cnn_jtl110gpu_to_jtl311linux:p75` | `pure_gpu` | 0.75 | 3.32354 | 2.60146 | 0.31513 | false |
| `migrate:cpu_node003_full_to_node005_empty:p25` | `pure_cpu` | 0.25 | 9.62908 | 23.508 | 4.40698 | false |
| `migrate:cpu_node003_full_to_node005_empty:p50` | `pure_cpu` | 0.50 | 9.62908 | 23.508 | 1.96752 | false |
| `migrate:cpu_node003_full_to_node005_empty:p75` | `pure_cpu` | 0.75 | 9.62908 | 23.508 | 0.810051 | false |
| `migrate:cpu_node003_half_to_node005_empty:p25` | `pure_cpu` | 0.25 | 17.833 | 23.508 | 4.40698 | false |
| `migrate:cpu_node003_half_to_node005_empty:p50` | `pure_cpu` | 0.50 | 17.833 | 23.508 | 1.96752 | false |
| `migrate:cpu_node003_half_to_node005_empty:p75` | `pure_cpu` | 0.75 | 17.833 | 23.508 | 0.810051 | false |
| `migrate:resac_ant_jtl110gpu_to_jtl311linux:p25` | `hybrid_rl` | 0.25 | 0.059437 | 0.0355299 | 0.0346744 | false |
| `migrate:resac_ant_jtl110gpu_to_jtl311linux:p50` | `hybrid_rl` | 0.50 | 0.059437 | 0.0355299 | 0.0342598 | false |
| `migrate:resac_ant_jtl110gpu_to_jtl311linux:p75` | `hybrid_rl` | 0.75 | 0.059437 | 0.0355299 | 0.033087 | false |

This certificate covers only controlled benchmark migrations with verified checkpoint/resume measurements at 25, 50, and 75 percent progress. Rates are exact hardware/load-state per-task lower service from the bound final cache. Ordinary running user tasks are excluded.
