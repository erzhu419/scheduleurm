# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_OPEN`
- Allow launch: `true`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl110_cnn_half_loaded_add_cnn` | `jtl110gpu` | `half_loaded` | `gpu_cnn_torch_resnet50` | 0 | false | `add_probe_returncode_127` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
