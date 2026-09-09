# Critical GPU LCB to service-cache-v2 merge gate

- Status: `WAIT_GPU_GATES`
- Complete cache ready: `False`
- Final cache written: `False`
- Base cache: `/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/service_cache_v2_critical_phase_20260802.json`
- Required protocol: `critical_gpu_phase_completion_v2`
- Inserted exact rows: `0` / `27`
- Legacy two-key index unchanged: `False`

## Representative hardware gates

| node | status | v2 PASS | source |
|---|---|:---:|---|
| `gpu_a` | `MISSING` | no | `/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_gpu_a_20260803.json` |
| `node007` | `MISSING` | no | `/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_node007_20260803.json` |
| `cpu_a` | `MISSING` | no | `/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_cpu_a_20260803.json` |

## Wait reasons

- `MISSING_GPU_GATE` node=gpu_a: /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_gpu_a_20260803.json
- `MISSING_GPU_GATE` node=node007: /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_node007_20260803.json
- `MISSING_GPU_GATE` node=cpu_a: /home/researcher/mine_code/scheduleurm/md/experiment_artifacts/critical_gpu_stochastic_lcb_gate_v2_cpu_a_20260803.json

The final cache exists only when gpu_a, node007, and cpu_a each provide a hardware-local v2 PASS certificate for all nine pre-registered empty-state GPU actions. Rows are exact workload_env x node_bucket x resource_state x profile records; they never update the legacy workload/profile fallback.
