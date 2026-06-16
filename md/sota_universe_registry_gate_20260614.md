# SOTA Universe Registry Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `registered_system_count` | 12 |
| `named_direct_runtime_probe_ready_count` | 5 |
| `named_direct_native_better_ready_count` | 5 |
| `registered_gpu_comparable_ready_count` | 5 |
| `registered_gpu_comparable_count` | 11 |
| `named_five_runtime_probe_ready` | true |
| `named_five_fullstack_superiority_ready` | false |
| `registered_sota_universe_superiority_ready` | false |
| `arbitrary_sota_superiority_ready` | false |

## Registry Rows

| System | Class | Repo | Runtime inventory | Entrypoint smoke | Same-workload runtime probe | Paired native-better | Comparable claim | Blocker |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Gavel | `named_runtime_probe` | true | false | false | true | true | true | none for the scoped same-host same-workload runtime-probe row |
| Pollux/AdaptDL | `named_runtime_probe` | true | false | false | true | true | true | none for the scoped same-host same-workload runtime-probe row |
| Sia | `named_runtime_probe` | true | false | false | true | true | true | none for the scoped same-host same-workload runtime-probe row |
| IADeep | `named_runtime_probe` | true | false | false | true | true | true | none for the scoped same-host same-workload runtime-probe row |
| Salus | `named_runtime_probe` | true | false | false | true | true | true | none for the scoped same-host same-workload runtime-probe row |
| Decima | `adjacent_simulator` | true | false | false | false | false | false | adjacent simulator domain; needs a Spark-DAG comparison protocol, not a GPU co-location runtime-probe row |
| Tiresias | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Python2 simulator is present, but the current host lacks the Python2 runtime dependencies needed for the entrypoint smoke, starting with numpy. |
| Themis | `paper_only_registered` | false | true | false | false | false | false | No official runtime repository was found in the current registry search; treat as paper-semantics baseline until an artifacted implementation is added. |
| Gandiva | `paper_only_registered` | false | true | false | false | false | false | No official public runtime repository was found in the current registry search; the system also relies on framework/runtime co-design, so direct comparison needs an artifacted implementation or faithful reimplementation protocol. |
| Shockwave | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Shockwave public runtime is cloned, but current-host smoke is blocked by generated protobuf stubs and/or Gurobi dependency. |
| AlloX | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | AlloX pointer repo plus Kubernetes fork and simulator are cloned. Java is present when the java probe passes, but the shipped simulator bin is incomplete on this host and javac is unavailable to rebuild the missing inner-class outputs; the run script also expects experiment arguments. |
| Optimus | `registered_not_yet_fullstack_compared` | true | true | false | false | false | false | Optimus public runtime is cloned, but current-host smoke is blocked by Python2 dependencies such as numpy and jinja2; full comparison also needs Kubernetes/MXNet wiring. |

## Scope

The named five runtime-probe claim is ready when Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus have scoped same-host same-workload rows with paired native-better evidence.  The registered-universe claim additionally requires comparable adapter rows for every registered GPU/DL scheduler.  The unbounded 'arbitrary SOTA' claim is not a finite experimental statement.
