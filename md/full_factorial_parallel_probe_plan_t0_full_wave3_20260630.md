# Full-Factorial Parallel Probe Plan

- Status: `PARALLEL_PROBE_PLAN_READY`
- Pending rows considered: `22`
- Selected rows: `0`
- Max rows per lane: `1`

## Resource Lock Policy

At most one active probe per physical node.  A node-level lock covers both CPU and GPU resources, so CPU resident probes never overlap with GPU ETA probes on the same host.  Homogeneous equivalent nodes are used only for equivalence validation unless explicitly selected as a spillover lane.

## Lanes

| Lane | Node lock | Resource lock | Selected | Commands |
|---|---|---|---:|---:|
| `gpu_jtl110_representative` | `jtl110gpu` | `jtl110gpu:gpu` | 0 | 0 |
| `gpu_jtl110_spillover` | `jtl110gpu2` | `jtl110gpu2:gpu` | 0 | 0 |
| `gpu_jtl311_cpu_fast` | `jtl311linux` | `jtl311linux:gpu` | 0 | 0 |
| `gpu_node007_direct` | `node007-direct` | `node007-direct:gpu` | 0 | 0 |
| `cpu_hpc_192c_representative` | `node001` | `node001:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node002` | `node002` | `node002:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node004` | `node004` | `node004:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node005` | `node005` | `node005:cpu` | 0 | 0 |
| `cpu_hpc_192c_spillover_node006` | `node006` | `node006:cpu` | 0 | 0 |
| `cpu_hpc_192c_equivalence` | `node003` | `node003:cpu` | 0 | 0 |
| `cpu_jtl110_representative` | `jtl110cpu` | `jtl110cpu:cpu` | 0 | 0 |
| `cpu_jtl110_spillover` | `jtl110cpu2` | `jtl110cpu2:cpu` | 0 | 0 |
| `cpu_node007_jtl110_family` | `node007-direct` | `node007-direct:cpu` | 0 | 0 |
| `cpu_jtl311_host` | `jtl311linux` | `jtl311linux:cpu` | 0 | 0 |

## Selected Rows

| Lane | Workload | Node | State | Missing profiles | Command |
|---|---|---|---|---|---|

## Claim Boundary

This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service cache only after their runner writes task-native tqdm/progress stable-rate summaries.
