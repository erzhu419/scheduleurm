# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_OPEN`
- Allow launch: `true`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu_gpu_llm_distilgpt2_llm_cpu_resident_cpu_worker_resident_p1` | `jtl110gpu` | `cpu_resident` | `gpu_llm_distilgpt2` | 0 | false | `stable_rate_not_ready` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
