# Service Cache v2 Live Merge Gate

- Status: `SERVICE_CACHE_V2_LIVE_MERGED`
- Pass: `true`
- Admitted rows: `34`
- Boundary rows: `1`
- Missing rows: `0`
- Workload envs: `ant, cnn, cpu, halfcheetah, hopper, llm, walker2d`
- Resource states: `cpu_resident, empty, full_loaded, half_loaded, high_vram_resident, mixed_colocation`

| Workload | Env | Node bucket | State | Semantics | Rate | Valid |
|---|---|---|---|---|---:|---:|
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 49.3363 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 50.0441 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `empty_profile_aggregate` | 26.5013 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `empty_profile_aggregate` | 23.6441 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `node007-direct:gpu_node007_4x12gb` | `empty` | `empty_profile_aggregate` | 126.909 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `node007-direct:gpu_node007_4x12gb` | `empty` | `empty_profile_aggregate` | 115.983 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `node007-direct:gpu_node007_4x12gb` | `empty` | `empty_profile_aggregate` | 114.339 | true |
| `gpu_llm_distilgpt2` | `llm` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 575.089 | true |
| `gpu_llm_distilgpt2` | `llm` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 1264.17 | true |
| `gpu_llm_distilgpt2` | `llm` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `empty_profile_aggregate` | 440.197 | true |
| `hybrid_rl_resac_ant` | `ant` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 0.70197 | true |
| `hybrid_rl_resac_ant` | `ant` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 0.544996 | true |
| `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 0.37037 | true |
| `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `empty_profile_aggregate` | 0.2882 | true |
| `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `empty_profile_aggregate` | 0.202217 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `half_loaded` | `marginal_add_one` | 15.2028 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `high_vram_resident` | `marginal_add_one` | 28.9301 | true |
| `cpu_heavy_local_bench` | `cpu` | `node001:cpu_hpc_192c` | `cpu_resident` | `marginal_add_one` | 12.2878 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `half_loaded` | `marginal_add_one` | 2.62253 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `full_loaded` | `marginal_add_one` | 28.1021 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `mixed_colocation` | `marginal_add_one` | 28.9391 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu:gpu_3080ti_12gb_dual` | `mixed_colocation` | `marginal_add_one` | 20.6049 | true |
| `cpu_heavy_local_bench` | `cpu` | `node003:cpu_hpc_192c` | `cpu_resident` | `marginal_add_one` | 12.5307 | true |
| `cpu_heavy_local_bench` | `cpu` | `node005:cpu_hpc_192c` | `cpu_resident` | `marginal_add_one` | 16.3924 | true |
| `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_ant` | `ant` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_hopper` | `hopper` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_walker2d` | `walker2d` | `jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_hopper` | `hopper` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_walker2d` | `walker2d` | `jtl110gpu:gpu_3080ti_12gb_dual` | `empty` | `None` | 0 | true |
| `gpu_llm_distilgpt2` | `llm` | `jtl110gpu2:gpu_3080ti_12gb_dual` | `empty` | `None` | 0 | true |
| `gpu_cnn_torch_resnet50` | `cnn` | `jtl110gpu2:gpu_3080ti_12gb_dual` | `empty` | `None` | 0 | true |
| `gpu_llm_distilgpt2` | `llm` | `node007-direct:gpu_node007_4x12gb` | `empty` | `None` | 0 | true |
| `hybrid_rl_resac_ant` | `ant` | `node007-direct:gpu_node007_4x12gb` | `empty` | `None` | 0 | true |

Merged theorem-facing live rows use task-native progress and stable-rate gates. They cover measured empty and selected under-load states only; they do not certify all future workloads, all mixed co-location states, or legacy scheduler hard-rule behavior.
