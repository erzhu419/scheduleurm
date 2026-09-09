# ETA Server Coverage Gate

- Status: `ETA_SERVER_COVERAGE_REQUIRED_READY`
- Pass: `true`
- Measured rows: `25`
- Partial rows: `0`
- Missing rows: `0`
- Required missing rows: `0`

| Node bucket | Workload | Env | State | Required profiles | Present | Missing | Requirement | Status |
|---|---|---|---|---:|---:|---:|---|---|
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[2]` | `[2]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `full_loaded` | `[4]` | `[4]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `high_vram_resident` | `[1]` | `[1]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `mixed_colocation` | `[2, 3]` | `[2, 3]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `gpu_llm_distilgpt2` | `llm` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `hybrid_rl_resac_ant` | `ant` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `hybrid_rl_resac_hopper` | `hopper` | `empty` | `[1]` | `[1]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu:gpu_3080ti_12gb_dual` | `hybrid_rl_resac_walker2d` | `walker2d` | `empty` | `[1]` | `[1]` | `[]` | `representative_full` | `measured` |
| `jtl110gpu2:gpu_3080ti_12gb_dual` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `homogeneous_equivalence` | `measured` |
| `jtl110gpu2:gpu_3080ti_12gb_dual` | `gpu_llm_distilgpt2` | `llm` | `empty` | `[1]` | `[1]` | `[]` | `homogeneous_equivalence` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `node_specific` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[2]` | `[2]` | `[]` | `node_specific` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `gpu_llm_distilgpt2` | `llm` | `empty` | `[1]` | `[1]` | `[]` | `node_specific` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `hybrid_rl_resac_ant` | `ant` | `empty` | `[1]` | `[1]` | `[]` | `node_specific` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `empty` | `[1, 2]` | `[1, 2]` | `[]` | `node_specific` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `hybrid_rl_resac_hopper` | `hopper` | `empty` | `[1]` | `[1]` | `[]` | `node_specific_pending` | `measured` |
| `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `hybrid_rl_resac_walker2d` | `walker2d` | `empty` | `[1]` | `[1]` | `[]` | `node_specific_pending` | `measured` |
| `node007-direct:gpu_node007_4x12gb` | `gpu_cnn_torch_resnet50` | `cnn` | `empty` | `[1, 2, 4]` | `[1, 2, 4]` | `[]` | `node_specific` | `measured` |
| `node007-direct:gpu_node007_4x12gb` | `gpu_llm_distilgpt2` | `llm` | `empty` | `[1]` | `[1]` | `[]` | `node_specific_pending` | `measured` |
| `node007-direct:gpu_node007_4x12gb` | `hybrid_rl_resac_ant` | `ant` | `empty` | `[1]` | `[1]` | `[]` | `node_specific_pending` | `measured` |
| `node001:cpu_hpc_192c` | `cpu_heavy_local_bench` | `cpu` | `cpu_resident` | `[1]` | `[1]` | `[]` | `cpu_representative` | `measured` |
| `node003:cpu_hpc_192c` | `cpu_heavy_local_bench` | `cpu` | `cpu_resident` | `[1]` | `[1]` | `[]` | `cpu_representative` | `measured` |
| `node005:cpu_hpc_192c` | `cpu_heavy_local_bench` | `cpu` | `cpu_resident` | `[1]` | `[1]` | `[]` | `cpu_equivalence` | `measured` |

This gate audits measured ETA coverage.  A measured representative or equivalence row does not imply every hardware/workload/load-state combination is globally certified.
