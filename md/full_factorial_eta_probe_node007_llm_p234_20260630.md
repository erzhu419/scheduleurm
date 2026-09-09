# Full-Factorial ETA Probe Runner

- Status: `PROBE_RUNNER_LAUNCHED`
- Allow launch: `true`
- Selected rows: `1`
- Launched rows: `1`
- Admitted rows: `0`
- Boundary rows: `1`

| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |
|---|---|---|---|---|---:|---:|---:|---|
| `gpu_node007_4x11gb|q01_low_cpu_high_gpu|gpu_llm_distilgpt2|llm|empty|none` | `node007-direct` | `gpu_llm_distilgpt2` | `llm` | `empty` | `[2, 3, 4]` | false | `[4]` | `p2=2975.96, p3=3998.75, p4=4735.53` |

Controlled full-factorial ETA probes.  Launches bypass the legacy scheduler dispatch path and require task-native tqdm/progress.  Invalid profiles are retained as scoped capacity/instability boundaries for the exact node/workload/state.
