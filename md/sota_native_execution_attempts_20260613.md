# SOTA Native Execution Attempts

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `native_execution_ready_count` | 3 |
| `direct_full_stack_ready_count` | 0 |
| `direct_fullstack_sota_superiority_ready` | false |

## Rows

| Adapter | Native scope | Native ready | Direct full-stack ready | Blocker |
|---|---|---:|---:|---|
| `gavel_simulation` | isolated generated-jobs, native trace, and bounded same-trace simulator microbaseline | true | false | service-unit equivalence and live full-stack Gavel cluster execution are not certified |
| `pollux_adaptdl_scheduler` | local Pollux policy-layer pytest with compatibility paths, no Kubernetes cluster | true | false | local Pollux optimizer policy tests pass; direct full stack still requires Kubernetes/Docker/AdaptDLJob execution |
| `sia_goodput_scheduler` | official Sia simulator entrypoint probe, no AdaptDL/Kubernetes cluster | false | false | Sia official artifact is cloned, but simulator entrypoint needs the official cvxpy CBC/GLPK and pymoo environment; physical run also requires AdaptDL/Kubernetes |
| `decima_simulator` | local simulator entrypoint help | true | false | Decima is a Spark-DAG simulator, not a GPU co-location scheduler baseline for Scheduleurm |
| `salus_gpu_sharing` | local Python driver entrypoint probe | false | false | Salus driver requires old future-fstrings/TensorFlow/C++ runtime and server stack; direct full-stack run is not available here |

## Scope

Native execution ledger for cloned external repositories on the current host.  Passing native rows do not imply direct full-stack SOTA superiority unless direct_full_stack_ready is true.
