# Critical GPU LCB to service-cache-v2 merge gate

- Status: `WAIT_GPU_GATES`
- Complete cache ready: `False`
- Final cache written: `False`
- Base cache: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/service_cache_v2_critical_phase_20260802.json`
- Required protocol: `critical_gpu_phase_completion_v2`
- Inserted exact rows: `0` / `27`
- Legacy two-key index unchanged: `False`

## Representative hardware gates

| node | status | v2 PASS | source |
|---|---|:---:|---|
| `jtl110gpu` | `MISSING` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_jtl110gpu_20260803.json` |
| `node007` | `MISSING` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_node007_20260803.json` |
| `jtl311linux` | `MISSING` | no | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_jtl311linux_20260803.json` |

## Wait reasons

- `MISSING_GPU_GATE` node=jtl110gpu: /home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_jtl110gpu_20260803.json
- `MISSING_GPU_GATE` node=node007: /home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_node007_20260803.json
- `MISSING_GPU_GATE` node=jtl311linux: /home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_jtl311linux_20260803.json

The final cache exists only when jtl110gpu, node007, and jtl311linux each provide a hardware-local v2 PASS certificate for all nine pre-registered empty-state GPU actions. Rows are exact workload_env x node_bucket x resource_state x profile records; they never update the legacy workload/profile fallback.
