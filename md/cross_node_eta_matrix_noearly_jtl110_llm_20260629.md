# Cross-Node ETA Matrix Gate

- Status: `LAUNCHED_PASS`
- Allow launch: `true`
- Pass: `true`

| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |
|---|---|---|---|---:|---|---:|
| `gpu_3080ti_12gb_dual|llm_distilgpt2` | `jtl110gpu` | `gpu_llm_distilgpt2` | `llm` | `[1, 2]` | `tqdm/progress` | true |

## Coverage

- `node007_and_jtl311_separate`: `false`
- `jtl110_equivalence_planned`: `true`
- `env_specific_rl_planned`: `false`
- `history_fallback_excluded`: `true`
