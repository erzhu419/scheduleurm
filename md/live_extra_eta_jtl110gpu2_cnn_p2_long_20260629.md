# Live Extra ETA Measurements

- Status: `LIVE_EXTRA_ETA_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `1`
- Boundary rows: `0`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `jtl110gpu2_cnn_equiv_p2_long` | `jtl110gpu2` | `gpu_cnn_torch_resnet50` | `cnn` | `[2]` | true | `p2=106.922` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
