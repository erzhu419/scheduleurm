# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `3`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl110_cnn_half_loaded_add_cnn` | `jtl110gpu` | `half_loaded` | `gpu_cnn_torch_resnet50` | 15.2028 | true | `` |
| `jtl110_cnn_high_vram_resident_add_cnn` | `jtl110gpu` | `high_vram_resident` | `gpu_cnn_torch_resnet50` | 28.9301 | true | `` |
| `node001_cpu_half_resident_add_cpu` | `node001` | `cpu_resident` | `cpu_heavy_local_bench` | 12.2878 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
