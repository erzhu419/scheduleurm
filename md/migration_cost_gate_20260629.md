# Controlled Migration Cost Gate

- Status: `MIGRATION_ACTION_MODEL_PASS`
- Pass: `true`
- Allowed rows: `9`
- Blocked rows: `21`
- No-touch safety: `true`

| Action | Progress | Ready | Reason | Penalty |
|---|---:|---:|---|---:|
| `migrate|task=cnn_bench|from=node003|to=jtl311linux|p=0.25` | 0.25 | true | `` | 10.000 |
| `migrate|task=cnn_bench_expensive|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=cnn_bench|from=node003|to=jtl311linux|p=0.50` | 0.50 | true | `` | 10.000 |
| `migrate|task=cnn_bench_expensive|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=cnn_bench|from=node003|to=jtl311linux|p=0.75` | 0.75 | true | `` | 10.000 |
| `migrate|task=cnn_bench_expensive|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=resac_halfcheetah|from=node003|to=jtl311linux|p=0.25` | 0.25 | true | `` | 10.000 |
| `migrate|task=resac_halfcheetah_expensive|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=resac_halfcheetah|from=node003|to=jtl311linux|p=0.50` | 0.50 | true | `` | 10.000 |
| `migrate|task=resac_halfcheetah_expensive|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=resac_halfcheetah|from=node003|to=jtl311linux|p=0.75` | 0.75 | true | `` | 10.000 |
| `migrate|task=resac_halfcheetah_expensive|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=freqduet_cpu_surrogate|from=node003|to=jtl311linux|p=0.25` | 0.25 | true | `` | 10.000 |
| `migrate|task=freqduet_cpu_surrogate_expensive|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=freqduet_cpu_surrogate|from=node003|to=jtl311linux|p=0.50` | 0.50 | true | `` | 10.000 |
| `migrate|task=freqduet_cpu_surrogate_expensive|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=freqduet_cpu_surrogate|from=node003|to=jtl311linux|p=0.75` | 0.75 | true | `` | 10.000 |
| `migrate|task=freqduet_cpu_surrogate_expensive|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `migration_cost_not_recovered` | 145.000 |
| `migrate|task=user_running_job|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `not_controlled_benchmark` | 10.000 |
| `migrate|task=user_running_job_expensive|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `not_controlled_benchmark` | 145.000 |
| `migrate|task=user_running_job|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `not_controlled_benchmark` | 10.000 |
| `migrate|task=user_running_job_expensive|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `not_controlled_benchmark` | 145.000 |
| `migrate|task=user_running_job|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `not_controlled_benchmark` | 10.000 |
| `migrate|task=user_running_job_expensive|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `not_controlled_benchmark` | 145.000 |
| `migrate|task=missing_checkpoint|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `checkpoint_missing` | 10.000 |
| `migrate|task=missing_checkpoint_expensive|from=node003|to=jtl311linux|p=0.25` | 0.25 | false | `checkpoint_missing` | 145.000 |
| `migrate|task=missing_checkpoint|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `checkpoint_missing` | 10.000 |
| `migrate|task=missing_checkpoint_expensive|from=node003|to=jtl311linux|p=0.50` | 0.50 | false | `checkpoint_missing` | 145.000 |
| `migrate|task=missing_checkpoint|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `checkpoint_missing` | 10.000 |
| `migrate|task=missing_checkpoint_expensive|from=node003|to=jtl311linux|p=0.75` | 0.75 | false | `checkpoint_missing` | 145.000 |

Controlled benchmark migration rows only; ordinary production running tasks are excluded.
