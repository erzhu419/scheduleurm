# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `4`
- Launched rows: `4`
- Admitted rows: `0`
- Boundary rows: `4`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_resac_ant|ant|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `hybrid_rl_resac_ant` | `ant` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[5]` | `p1=0.951412, p2=1.15324, p3=1.25277, p4=1.43614, p5=0.90022, p6=0` |
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_resac_halfcheetah|halfcheetah|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `hybrid_rl_resac_halfcheetah` | `halfcheetah` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[6]` | `p1=0.511581, p2=0.485005, p3=0.482171, p4=0, p5=0, p6=0.159576` |
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_resac_hopper|hopper|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `hybrid_rl_resac_hopper` | `hopper` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[4]` | `p1=1.49534, p2=1.99328, p3=2.02844, p4=1.72076, p5=1.85594, p6=0.850997` |
| `gpu_node007_4x11gb|q11_high_cpu_high_gpu|hybrid_rl_resac_walker2d|walker2d|full_loaded|same_workload_to_capacity_boundary` | `node007-direct` | `hybrid_rl_resac_walker2d` | `walker2d` | `full_loaded` | `[1, 2, 3, 4, 5, 6]` | false | `[5]` | `p1=1.56558, p2=2.4195, p3=2.65264, p4=2.5818, p5=2.37553, p6=2.46218` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
