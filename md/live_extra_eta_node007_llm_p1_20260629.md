# Live Extra ETA Measurements

- Status: `LIVE_EXTRA_ETA_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `1`
- Boundary rows: `0`

| Spec | Node | Workload | Env | Profiles | Valid | Rates |
|---|---|---|---|---:|---:|---|
| `node007_llm_p1` | `node007-direct` | `gpu_llm_distilgpt2` | `llm` | `[1]` | true | `p1=1311.47` |

Controlled extra task-native ETA rows for equivalent hardware checks and node/env-aware RL service. Legacy scheduler hard limits are bypassed; ordinary user tasks are never launched or migrated by this gate.
