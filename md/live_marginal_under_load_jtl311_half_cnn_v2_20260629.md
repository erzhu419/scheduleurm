# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `1`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl311_cnn_half_loaded_add_cnn` | `jtl311linux` | `half_loaded` | `gpu_cnn_torch_resnet50` | 2.62253 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
