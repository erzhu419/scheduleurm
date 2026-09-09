# Live Extra ETA Measurements

- Status: `LIVE_EXTRA_ETA_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `0`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `jtl110gpu_resac_hopper_p1` | `jtl110gpu` | `hybrid_rl_resac_hopper` | `hopper` | `[1]` | true | `p1=1.11111` |
| `jtl110gpu_resac_walker2d_p1` | `jtl110gpu` | `hybrid_rl_resac_walker2d` | `walker2d` | `[1]` | true | `p1=1.17647` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
