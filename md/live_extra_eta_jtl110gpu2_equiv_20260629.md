# Live Extra ETA Measurements

- Status: `LIVE_EXTRA_ETA_OPEN`
- Allow launch: `true`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `2`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `jtl110gpu2_cnn_equiv` | `jtl110gpu2` | `gpu_cnn_torch_resnet50` | `cnn` | `[1, 2]` | false | `p1=0, p2=0` |
| `jtl110gpu2_llm_equiv` | `jtl110gpu2` | `gpu_llm_distilgpt2` | `llm` | `[1]` | false | `p1=0` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
