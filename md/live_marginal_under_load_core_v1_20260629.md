# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl311_cnn_half_loaded_add_cnn` | `jtl311linux` | `half_loaded` | `gpu_cnn_torch_resnet50` | 0 | false | `stable_rate_not_ready` |
| `jtl110_cnn_high_vram_resident_add_cnn` | `jtl110gpu` | `high_vram_resident` | `gpu_cnn_torch_resnet50` | 28.9231 | true | `` |
| `node001_cpu_half_resident_add_cpu` | `node001` | `cpu_resident` | `cpu_heavy_local_bench` | 11.1044 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
