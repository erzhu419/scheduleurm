# Direct SOTA Full-Stack Readiness

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `entrypoint_smoke_pass_count` | 3 |
| `same_workload_full_stack_ready_count` | 0 |
| `direct_full_stack_same_workload_ready` | false |
| `policy_semantics_fallback_available` | true |

## Tool Inventory

| Tool | Available | Path |
|---|---:|---|
| `docker` | false | `` |
| `go` | false | `` |
| `kubectl` | false | `` |
| `nvidia-smi` | true | `/usr/lib/wsl/lib/nvidia-smi` |
| `python3` | true | `/usr/bin/python3` |

## Adapter Readiness

| Adapter | Repo | Smoke | Native generated | Native trace | Native microbaseline | Missing tools | Missing workload assets | Full-stack ready | Blockers |
|---|---:|---:|---:|---|---:|---|---|---:|---|
| `gavel_simulation` | true | true | true | `PASS` | true | none | none | false | Gavel native trace and Scheduleurm trace-seed smoke pass, but service-unit equivalence is not yet a direct performance baseline; Gavel native simulator microbaseline rows exist, but they still do not certify Scheduleurm measured-service-unit equivalence; Scheduleurm trace/throughput seeds exist, but same-workload Gavel schema/service-unit validation is still required |
| `pollux_adaptdl_scheduler` | true | true | false | `` | false | `kubectl`, `docker` | none | false | missing required tools: kubectl, docker; AdaptDLJob manifest seeds exist, but a Kubernetes cluster, container image, and isolated test namespace are still required |
| `sia_goodput_scheduler` | true | false | false | `` | false | `kubectl`, `docker` | none | false | missing required tools: kubectl, docker; Sia official artifact is cloned, but simulator execution needs the official cvxpy CBC/GLPK and pymoo environment; physical-cluster execution needs AdaptDL on Kubernetes with container images and cluster-specific GPU-type mapping |
| `iadeep_kubernetes_extender` | true | false | false | `` | false | `go`, `kubectl`, `docker` | none | false | missing required tools: go, kubectl, docker; IADeep pod manifest seeds exist, but a Kubernetes cluster with NVIDIA device-plugin/runtime wiring is still required |
| `salus_gpu_sharing` | true | false | false | `` | false | `docker` | none | false | missing required tools: docker; Salus benchmark seed exists, but Salus server/runtime execution is still required |
| `decima_simulator` | true | true | false | `` | false | none | none | false | Decima is a Spark-DAG simulator, so even a runnable entrypoint is not a GPU co-location system baseline |

## Scope

read-only readiness audit for direct external-system baselines. It certifies local repo/entrypoint/stack blockers; it does not claim direct full-stack superiority unless direct_full_stack_same_workload_ready is true.
