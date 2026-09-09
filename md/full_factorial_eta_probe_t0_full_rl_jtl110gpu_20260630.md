# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `5`
- Launched rows: `5`
- Admitted rows: `3`
- Boundary rows: `0`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_ant|ant|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_resac_ant` | `ant` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[]` | `p1=0.560228, p2=0.547225, p3=0.550502, p4=0.889795, p5=0.772581, p6=0` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_halfcheetah|halfcheetah|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | true | `[]` | `p1=0.295437, p2=0.2874, p3=0.286619, p4=0.286177, p5=0.28818, p6=0.413331` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_hopper|hopper|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_resac_hopper` | `hopper` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | true | `[]` | `p1=0.909091, p2=1.12026, p3=0.894554, p4=0.895271, p5=1.54628, p6=1.47805` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_resac_walker2d|walker2d|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_resac_walker2d` | `walker2d` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | true | `[]` | `p1=0.952467, p2=0.915385, p3=1.07549, p4=1.03751, p5=1.80906, p6=1.82098` |
| `gpu_3080ti_12gb_dual|q11_high_cpu_high_gpu|hybrid_rl_bapr_ant|bapr_ant|full_loaded|same_workload_to_capacity_boundary` | `jtl110gpu` | `hybrid_rl_bapr_ant` | `bapr_ant` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[]` | `p1=0.348436, p2=0.435769, p3=0.418524, p4=0, p5=0.667715, p6=0.625045` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
