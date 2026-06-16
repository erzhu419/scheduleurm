# Multi-Node Theorem Shadow Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `MULTINODE_THEOREM_SHADOW_CLOSED` |
| `probe_remotes` | true |
| `reachable_node_count` | 2 |
| `reachable_gpu_node_count` | 2 |
| `reachable_cpu_node_count` | 2 |
| `candidate_row_count` | 18 |
| `theorem_candidate_row_count` | 18 |
| `candidate_configuration_count` | 626 |
| `cross_node_candidate_family_ready` | true |
| `cross_node_selected_configuration_ready` | true |
| `multinode_theorem_shadow_ready` | true |
| `multinode_original_launch_claim_ready` | false |

## Selected Actions

`shadow-q01-llm|jtl110gpu:gpu:0|p1`, `shadow-q11-hybrid|jtl110gpu:gpu:1|p1`, `shadow-q01-gpu-heavy|jtl110gpu2:gpu:1|p1`, `shadow-q01-cnn|jtl110gpu2:gpu:0|p1`, `shadow-q00-light|jtl110gpu2:cpu|p1`

## Node Inventory

| Node | Probed | Reachable | CPUs | GPUs | Note |
|---|---:|---:|---:|---:|---|
| `jtl110gpu` | true | true | `24` | `2` |  |
| `jtl110gpu2` | true | true | `24` | `2` |  |
| `node001` | true | false | `None` | `None` | ssh: Could not resolve hostname node001: Temporary failure in name resolution  |
| `node002` | true | false | `None` | `None` | ssh: Could not resolve hostname node002: Temporary failure in name resolution  |
| `node003` | true | false | `None` | `None` | ssh: Could not resolve hostname node003: Temporary failure in name resolution  |
| `node004` | true | false | `None` | `None` | ssh: Could not resolve hostname node004: Temporary failure in name resolution  |
| `node005` | true | false | `None` | `None` | ssh: Could not resolve hostname node005: Temporary failure in name resolution  |
| `node006` | true | false | `None` | `None` | ssh: Could not resolve hostname node006: Temporary failure in name resolution  |
| `node007` | true | false | `None` | `None` | ssh: Could not resolve hostname node007: Temporary failure in name resolution  |

## Blocker

none

## Scope

Read-only cross-node theorem candidate-family shadow.  It shows that certified lower-service candidates can be enumerated and globally scored across visible nodes.  It is not an original multi-node full-stack launched-completion comparison.
