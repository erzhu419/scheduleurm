# Live Extra ETA Measurements

- Status: `LIVE_EXTRA_ETA_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `0`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `jtl311_resac_halfcheetah_p1` | `jtl311linux` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `[1]` | true | `p1=0.28612` |
| `jtl311_resac_ant_p1` | `jtl311linux` | `hybrid_rl_resac_ant` | `ant` | `[1]` | true | `p1=0.501253` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
