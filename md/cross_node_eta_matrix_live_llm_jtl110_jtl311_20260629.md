# Cross-Node ETA Matrix Gate

- Status: `LAUNCHED_OPEN`
- Allow launch: `true`
- Pass: `false`

| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |
|---|---|---|---|---:|---|---:|
| `gpu_3080ti_12gb_dual|llm_distilgpt2` | `jtl110gpu` | `gpu_llm_distilgpt2` | `llm` | `[1, 2]` | `tqdm/progress` | true |
| `gpu_rtx2080_8gb_dual_cpu_fast|llm_distilgpt2` | `jtl311linux` | `gpu_llm_distilgpt2` | `llm` | `[1]` | `tqdm/progress` | true |

## Coverage

- `node007_and_jtl311_separate`: `false`
- `jtl110_equivalence_planned`: `true`
- `env_specific_rl_planned`: `false`
- `history_fallback_excluded`: `true`
