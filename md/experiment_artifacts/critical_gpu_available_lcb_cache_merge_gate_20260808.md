# Critical GPU LCB to service-cache-v2 merge gate

- Status: `PASS`
- Complete cache ready: `False`
- Final cache written: `True`
- Base cache: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/service_cache_v2_critical_phase_20260802.json`
- Required protocol: `None`
- Inserted exact rows: `27` / `None`
- Legacy two-key index unchanged: `True`

## Representative hardware gates

| node | status | protocol PASS | source |
|---|---|:---:|---|
| `jtl110gpu` | `PASS` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v8_jtl110gpu_20260803.json` |
| `jtl110gpu2` | `PASS` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v8_jtl110gpu2_20260803.json` |
| `node007` | `PASS` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v9_node007_20260808.json` |

This cache covers only the nine declared empty-state actions on jtl110gpu, jtl110gpu2, and node007. The two RTX 3080 Ti hosts are kept as separate operational execution classes because the pre-registered equivalence gate failed; no cross-host service rows are pooled. node007 uses the p10 transport-corrected composite certificate. jtl311linux remains pending and is neither imputed nor represented by another host. The cache does not cover loaded states, unmeasured profiles, or arbitrary future workloads.
