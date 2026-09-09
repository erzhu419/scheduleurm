# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl110_cnn_full_loaded_add_cnn` | `jtl110gpu` | `full_loaded` | `gpu_cnn_torch_resnet50` | 28.1021 | true | `` |
| `jtl110_llm_resident_add_cnn` | `jtl110gpu` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 28.9391 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
