# Multi-Node Original Deployment Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `probe_remotes` | false |
| `same_host_named_runtime_probe_ready` | true |
| `same_host_named_fullstack_ready` | false |
| `reachable_node_count` | 0 |
| `reachable_gpu_node_count` | 0 |
| `reachable_cpu_node_count` | 0 |
| `infrastructure_two_node_visible` | false |
| `gpu_two_node_visible` | false |
| `scheduleurm_multinode_history_completion_ready` | true |
| `original_multinode_deployment_rows_ready` | true |
| `multinode_original_deployment_superiority_ready` | false |

## Node Inventory

| Node | Probed | Reachable | CPUs | GPUs | Note |
|---|---:|---:|---:|---:|---|
| `jtl110gpu` | false | NA | `None` | `None` | probe disabled |
| `jtl110gpu2` | false | NA | `None` | `None` | probe disabled |
| `node001` | false | NA | `None` | `None` | probe disabled |
| `node002` | false | NA | `None` | `None` | probe disabled |
| `node003` | false | NA | `None` | `None` | probe disabled |
| `node004` | false | NA | `None` | `None` | probe disabled |
| `node005` | false | NA | `None` | `None` | probe disabled |
| `node006` | false | NA | `None` | `None` | probe disabled |
| `node007` | false | NA | `None` | `None` | probe disabled |

## Scope

Scheduleurm-native multi-node launched/completion history is certified separately from external SOTA original-deployment superiority.  Same-host external runtime-probe rows and Scheduleurm multi-node history do not imply that every external scheduler has been run through its original multi-node worker/control-plane path.
