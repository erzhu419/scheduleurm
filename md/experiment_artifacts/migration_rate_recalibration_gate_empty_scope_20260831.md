# Migration Rate Recalibration Gate

- Status: `PASS`
- Pass: `true`
- Final cache: `md/experiment_artifacts/service_cache_v2_all_hardware_gpu_phase_20260809.json`
- Ready rows: `12` / `12`
- Ordinary user tasks excluded: `true`
- Old rates/flags inherited: `false`

| Migration | Progress | Ready | Old rate | New rate | Keep (s) | Migrate (s) | Net (s) | Beneficial | Block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cnn_jtl110gpu_to_jtl311linux` | 0.25 | true | 3.323542 | 2.601464 | 13.539773 | 59.012550 | -45.472777 | false | `` |
| `cnn_jtl110gpu_to_jtl311linux` | 0.50 | true | 3.323542 | 2.601464 | 9.026515 | 53.191396 | -44.164881 | false | `` |
| `cnn_jtl110gpu_to_jtl311linux` | 0.75 | true | 3.323542 | 2.601464 | 4.513258 | 47.599476 | -43.086218 | false | `` |
| `resac_ant_jtl110gpu_to_jtl311linux` | 0.25 | true | 0.059437 | 0.035530 | 504.735684 | 865.191958 | -360.456274 | false | `` |
| `resac_ant_jtl110gpu_to_jtl311linux` | 0.50 | true | 0.059437 | 0.035530 | 336.490456 | 583.773950 | -247.283494 | false | `` |
| `resac_ant_jtl110gpu_to_jtl311linux` | 0.75 | true | 0.059437 | 0.035530 | 168.245228 | 302.233072 | -133.987844 | false | `` |
| `cpu_node003_half_to_node005_empty` | 0.25 | true | 17.832984 | 23.508016 | 6.729104 | 27.229519 | -20.500415 | false | `` |
| `cpu_node003_half_to_node005_empty` | 0.50 | true | 17.832984 | 23.508016 | 4.486069 | 40.660297 | -36.174228 | false | `` |
| `cpu_node003_half_to_node005_empty` | 0.75 | true | 17.832984 | 23.508016 | 2.243035 | 49.379584 | -47.136550 | false | `` |
| `cpu_node003_full_to_node005_empty` | 0.25 | true | 9.629082 | 23.508016 | 12.462247 | 27.229519 | -14.767272 | false | `` |
| `cpu_node003_full_to_node005_empty` | 0.50 | true | 9.629082 | 23.508016 | 8.308165 | 40.660297 | -32.352132 | false | `` |
| `cpu_node003_full_to_node005_empty` | 0.75 | true | 9.629082 | 23.508016 | 4.154082 | 49.379584 | -45.225502 | false | `` |

This certificate recalculates controlled benchmark migration actions for the declared exact hardware/load states. It does not authorize migration of ordinary user tasks or extrapolate to missing states.
