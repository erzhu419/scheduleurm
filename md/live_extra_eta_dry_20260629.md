# Live Extra ETA Measurements

- Status: `DRY_RUN_READY`
- Allow launch: `false`
- Pass: `true`
- Admitted rows: `0`
- Boundary rows: `0`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `jtl110gpu2_cnn_equiv` | `jtl110gpu2` | `gpu_cnn_torch_resnet50` | `cnn` | `[1, 2]` | false | `` |
| `jtl311_resac_halfcheetah_p1` | `jtl311linux` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `[1]` | false | `` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
