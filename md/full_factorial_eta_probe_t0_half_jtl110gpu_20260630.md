# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `8`
- Launched rows: `8`
- Admitted rows: `7`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_cnn_torch_resnet50|cnn|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `gpu_cnn_torch_resnet50` | `cnn` | `half_loaded` | `[3, 4, 8]` | false | `[4]` | `p3=76.6722, p4=92.7671, p8=0` |
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `gpu_llm_distilgpt2` | `llm` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=1181.92, p3=1754.69, p4=2228.95` |
| `gpu_3080ti_12gb_dual|q01_low_cpu_high_gpu|gpu_heavy_jax_matmul|gpu_matmul|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `gpu_heavy_jax_matmul` | `gpu_matmul` | `half_loaded` | `[2, 3, 4, 8]` | true | `[]` | `p2=3558.17, p3=5292.86, p4=7175.05, p8=13336.7` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_ant|ant|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `hybrid_rl_resac_ant` | `ant` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=0.553003, p3=0.549236, p4=0.757925` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_halfcheetah|halfcheetah|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=0.289604, p3=0.286009, p4=0.286061` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_hopper|hopper|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `hybrid_rl_resac_hopper` | `hopper` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=0.892974, p3=0.898853, p4=1.1082` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_walker2d|walker2d|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `hybrid_rl_resac_walker2d` | `walker2d` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=1.16219, p3=0.929122, p4=1.37121` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|half_loaded|same_workload_half_capacity` | `jtl110gpu` | `hybrid_rl_bapr_ant` | `bapr_ant` | `half_loaded` | `[2, 3, 4]` | true | `[]` | `p2=0.509021, p3=0.452484, p4=0.452971` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
