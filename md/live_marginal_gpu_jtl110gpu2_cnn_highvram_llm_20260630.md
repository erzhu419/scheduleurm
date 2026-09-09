# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `jtl110gpu2_cnn_high_vram_resident_p2_add_cnn` | `jtl110gpu2` | `high_vram_resident` | `gpu_cnn_torch_resnet50` | 55.9285 | true | `` |
| `jtl110gpu2_cnn_high_vram_resident_p3_add_cnn` | `jtl110gpu2` | `high_vram_resident` | `gpu_cnn_torch_resnet50` | 55.8442 | false | `background_startup_failed` |
| `jtl110gpu2_llm_resident_add_cnn_p1` | `jtl110gpu2` | `mixed_colocation` | `gpu_cnn_torch_resnet50` | 44.7753 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
