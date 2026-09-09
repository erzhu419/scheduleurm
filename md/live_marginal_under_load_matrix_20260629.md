# Live Marginal Under-Load Matrix

- Status: `DRY_RUN_READY`
- Allow launch: `false`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl110_cnn_half_loaded_add_cnn` | `jtl110gpu` | `half_loaded` | `gpu_cnn_torch_resnet50` | 0 | false | `dry_run_manifest` |
| `jtl311_cnn_half_loaded_add_cnn` | `jtl311linux` | `half_loaded` | `gpu_cnn_torch_resnet50` | 0 | false | `dry_run_manifest` |
| `jtl110_cnn_high_vram_resident_add_cnn` | `jtl110gpu` | `high_vram_resident` | `gpu_cnn_torch_resnet50` | 0 | false | `dry_run_manifest` |
| `node001_cpu_half_resident_add_cpu` | `node001` | `cpu_resident` | `cpu_heavy_local_bench` | 0 | false | `dry_run_manifest` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
