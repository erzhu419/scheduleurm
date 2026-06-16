# Registered SOTA Runtime Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `REGISTERED_SOTA_RUNTIME_INVENTORY_READY_FULLSTACK_PENDING` |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `registered_extension_system_count` | 6 |
| `runnable_public_runtime_system_count` | 4 |
| `entrypoint_smoke_ready_count` | 0 |
| `same_workload_fullstack_ready_count` | 0 |

## Rows

| System | Public runtime | Repo/Paper inventory | Entrypoint smoke | Full-stack row | Blocker |
|---|---:|---:|---:|---:|---|
| Tiresias | true | true | false | false | Python2 simulator is present, but the current host lacks the Python2 runtime dependencies needed for the entrypoint smoke, starting with numpy. |
| Shockwave | true | true | false | false | Shockwave public runtime is cloned, but current-host smoke is blocked by generated protobuf stubs and/or Gurobi dependency. |
| AlloX | true | true | false | false | AlloX pointer repo plus Kubernetes fork and simulator are cloned. Java is present when the java probe passes, but the shipped simulator bin is incomplete on this host and javac is unavailable to rebuild the missing inner-class outputs; the run script also expects experiment arguments. |
| Optimus | true | true | false | false | Optimus public runtime is cloned, but current-host smoke is blocked by Python2 dependencies such as numpy and jinja2; full comparison also needs Kubernetes/MXNet wiring. |
| Themis | false | true | false | false | No official runtime repository was found in the current registry search; treat as paper-semantics baseline until an artifacted implementation is added. |
| Gandiva | false | true | false | false | No official public runtime repository was found in the current registry search; the system also relies on framework/runtime co-design, so direct comparison needs an artifacted implementation or faithful reimplementation protocol. |

## Scope

Read-only runtime inventory for registered SOTA systems beyond the named five.  It distinguishes runnable public repositories from paper-only systems and records dependency blockers without changing the host environment.
