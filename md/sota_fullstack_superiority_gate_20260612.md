# SOTA Full-Stack Superiority Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `direct_fullstack_sota_superiority_ready` | false |
| `full_stack_ready_count` | 0 |
| `policy_semantics_comparison_ready` | true |
| `hard_blocker_certificate_ready` | true |

## Tool Blockers

| Tool | Available | Path / version |
|---|---:|---|
| `docker` | false | `` |
| `go` | false | `` |
| `kubectl` | false | `` |
| `nvidia-smi` | true | `/usr/lib/wsl/lib/nvidia-smi` |
| `python3` | true | `/usr/bin/python3` |

## System Rows

| Adapter | Smoke | Native microbaseline | Full-stack ready | Superiority allowed | Reason |
|---|---:|---:|---:|---:|---|
| `gavel_simulation` | true | true | false | false | native simulator microbaseline exists, but service-unit equivalence and full-stack production execution are not certified |
| `pollux_adaptdl_scheduler` | true | false | false | false | missing required tools: kubectl, docker; AdaptDLJob manifest seeds exist, but a Kubernetes cluster, container image, and isolated test namespace are still required |
| `iadeep_kubernetes_extender` | false | false | false | false | missing required tools: go, kubectl, docker; IADeep pod manifest seeds exist, but a Kubernetes cluster with NVIDIA device-plugin/runtime wiring is still required |
| `salus_gpu_sharing` | false | false | false | false | missing required tools: docker; Salus benchmark seed exists, but Salus server/runtime execution is still required |
| `decima_simulator` | true | false | false | false | Decima is a Spark-DAG simulator, so even a runnable entrypoint is not a GPU co-location system baseline |

## Scope

Strict gate for direct full-stack superiority claims.  It confirms the current package has policy-semantics replay, same-workload seeds, and Gavel native simulator microbaseline evidence, while rejecting direct full-stack superiority because the required external runtime stacks are not available or not service-unit equivalent.
