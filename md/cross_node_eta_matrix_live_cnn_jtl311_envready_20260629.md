# Cross-Node ETA Matrix Gate

- Status: `LAUNCHED_OPEN`
- Allow launch: `true`
- Pass: `false`

| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |
|---|---|---|---|---:|---|---:|
| `gpu_rtx2080_8gb_dual_cpu_fast|cnn_resnet50` | `jtl311linux` | `gpu_cnn_torch_resnet50` | `cnn` | `[1, 2, 4]` | `tqdm/progress` | true |

## Coverage

- `node007_and_jtl311_separate`: `false`
- `jtl110_equivalence_planned`: `false`
- `env_specific_rl_planned`: `false`
- `history_fallback_excluded`: `true`
