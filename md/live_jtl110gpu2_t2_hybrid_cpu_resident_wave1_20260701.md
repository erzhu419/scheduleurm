# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_DESIGN_OPEN`
- Allow launch: `true`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `2`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `design_jtl110gpu2_hybrid_rl_resac_hopper_hopper_cpu_resident_cpu_worker_resident_p1` | `jtl110gpu2` | `cpu_resident` | `hybrid_rl_resac_hopper` | 0 | false | `add_probe_returncode_139` |
| `design_jtl110gpu2_hybrid_rl_resac_walker2d_walker2d_cpu_resident_cpu_worker_resident_p1` | `jtl110gpu2` | `cpu_resident` | `hybrid_rl_resac_walker2d` | 0 | false | `add_probe_returncode_139` |

Generated from full-factorial T2 design rows. Each selected row uses a resident/mixed-load harness and task-native progress ETA.
