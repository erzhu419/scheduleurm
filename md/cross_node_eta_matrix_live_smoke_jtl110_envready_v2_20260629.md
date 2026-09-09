# Cross-Node ETA Matrix Gate

- Status: `LAUNCHED_OPEN`
- Allow launch: `true`
- Pass: `false`

| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |
|---|---|---|---|---:|---|---:|
| `gpu_3080ti_12gb_dual|cnn_resnet50` | `jtl110gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `[1, 2, 4]` | `tqdm/progress` | true |

## Coverage

- `node007_and_jtl311_separate`: `false`
- `jtl110_equivalence_planned`: `true`
- `env_specific_rl_planned`: `false`
- `history_fallback_excluded`: `true`
