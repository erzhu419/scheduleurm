# SOTA Universe Registry Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `registered_system_count` | 12 |
| `named_direct_fullstack_ready_count` | 5 |
| `named_direct_superiority_ready_count` | 5 |
| `registered_gpu_comparable_ready_count` | 5 |
| `registered_gpu_comparable_count` | 11 |
| `named_five_fullstack_superiority_ready` | true |
| `registered_sota_universe_superiority_ready` | false |
| `arbitrary_sota_superiority_ready` | false |

## Registry Rows

| System | Class | Repo | Runtime inventory | Entrypoint smoke | Same-workload full-stack | Paired superiority | Comparable claim | Blocker |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Gavel | `named_direct_fullstack` | true | false | false | true | true | true | none for the scoped same-host same-workload named row |
| Pollux/AdaptDL | `named_direct_fullstack` | true | false | false | true | true | true | none for the scoped same-host same-workload named row |
| Sia | `named_direct_fullstack` | true | false | false | true | true | true | none for the scoped same-host same-workload named row |
| IADeep | `named_direct_fullstack` | true | false | false | true | true | true | none for the scoped same-host same-workload named row |
| Salus | `named_direct_fullstack` | true | false | false | true | true | true | none for the scoped same-host same-workload named row |
| Decima | `adjacent_simulator` | true | false | false | false | false | false | adjacent simulator domain; needs a Spark-DAG comparison protocol, not a GPU co-location full-stack row |
| Tiresias | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Python2 simulator is present, but the current host lacks the Python2 runtime dependencies needed for the entrypoint smoke, starting with numpy. |
| Themis | `paper_only_registered` | false | true | false | false | false | false | No official runtime repository was found in the current registry search; treat as paper-semantics baseline until an artifacted implementation is added. |
| Gandiva | `paper_only_registered` | false | true | false | false | false | false | No official public runtime repository was found in the current registry search; the system also relies on framework/runtime co-design, so direct comparison needs an artifacted implementation or faithful reimplementation protocol. |
| Shockwave | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Shockwave public runtime is cloned, but current-host smoke is blocked by generated protobuf stubs and/or Gurobi dependency. |
| AlloX | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | AlloX pointer repo plus Kubernetes fork and simulator are cloned. Java is present when the java probe passes, but the shipped simulator bin is incomplete on this host and javac is unavailable to rebuild the missing inner-class outputs; the run script also expects experiment arguments. |
| Optimus | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Optimus public runtime is cloned, but current-host smoke is blocked by Python2 dependencies such as numpy and jinja2; full comparison also needs Kubernetes/MXNet wiring. |

## Scope

The named five direct full-stack claim is ready when Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus have same-host same-workload rows.  The registered-universe claim additionally requires comparable full-stack rows for every registered GPU/DL scheduler.  The unbounded 'arbitrary SOTA' claim is not a finite experimental statement.
