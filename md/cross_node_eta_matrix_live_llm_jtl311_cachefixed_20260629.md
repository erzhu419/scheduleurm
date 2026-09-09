# Cross-Node ETA Matrix Gate

- Status: `LAUNCHED_PASS`
- Allow launch: `true`
- Pass: `true`

| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |
|---|---|---|---|---:|---|---:|
| `gpu_rtx2080_8gb_dual_cpu_fast|llm_distilgpt2` | `jtl311linux` | `gpu_llm_distilgpt2` | `llm` | `[1]` | `tqdm/progress` | true |

## Coverage

- `node007_and_jtl311_separate`: `false`
- `jtl110_equivalence_planned`: `false`
- `env_specific_rl_planned`: `false`
- `history_fallback_excluded`: `true`
