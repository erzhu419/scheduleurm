# Registered SOTA Adapter Closure Gate

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `REGISTERED_SOTA_POLICY_ADAPTER_UNIVERSE_CLOSED_DIRECT_BINARY_PENDING` |
| `registered_system_count` | 11 |
| `policy_service_unit_adapter_ready_count` | 11 |
| `direct_external_binary_adapter_ready_count` | 5 |
| `registered_systems_without_any_policy_adapter_count` | 0 |
| `registered_systems_without_direct_binary_adapter_count` | 6 |
| `registered_policy_adapter_universe_ready` | true |
| `registered_direct_external_binary_superiority_ready` | false |

## Rows

| System | Policy/service-unit adapter | Direct binary adapter | Runtime inventory | Entrypoint smoke | Evidence | Blocker |
|---|---:|---:|---:|---:|---|---|
| Gavel | true | true | true | false | Named system direct full-stack row is closed in the named-five gate. | none for named same-host same-workload direct binary row |
| Pollux/AdaptDL | true | true | true | false | Named system direct full-stack row is closed in the named-five gate. | none for named same-host same-workload direct binary row |
| Sia | true | true | true | false | Named system direct full-stack row is closed in the named-five gate. | none for named same-host same-workload direct binary row |
| IADeep | true | true | true | false | Named system direct full-stack row is closed in the named-five gate. | none for named same-host same-workload direct binary row |
| Salus | true | true | true | false | Named system direct full-stack row is closed in the named-five gate. | none for named same-host same-workload direct binary row |
| Tiresias | true | false | true | false | Tiresias trace/cluster simulator files are present; Python2 numpy blocks executable smoke. | none for policy/service-unit adapter; direct binary row remains separate; Python2 simulator is present, but the current host lacks the Python2 runtime dependencies needed for the entrypoint smoke, starting with numpy. |
| Themis | true | false | true | false | Themis is represented by policy-semantics action family; no official runtime row is present in this package. | none for policy/service-unit adapter; direct binary row remains separate; No official runtime repository was found in the current registry search; treat as paper-semantics baseline until an artifacted implementation is added. |
| Gandiva | true | false | true | false | Gandiva is represented by policy-semantics action family; no official runtime row is present in this package. | none for policy/service-unit adapter; direct binary row remains separate; No official public runtime repository was found in the current registry search; the system also relies on framework/runtime co-design, so direct comparison needs an artifacted implementation or faithful reimplementation protocol. |
| Shockwave | true | false | true | false | Shockwave reproduce scripts/traces are present (4 scripts, 5 traces); solver/protobuf stack blocks executable same-workload smoke. | none for policy/service-unit adapter; direct binary row remains separate; Shockwave public runtime is cloned, but current-host smoke is blocked by generated protobuf stubs and/or Gurobi dependency. |
| AlloX | true | false | true | false | AlloX simulator output/log CSVs are present and parseable (6 files); Java classpath is incomplete for rerun on this host. | none for policy/service-unit adapter; direct binary row remains separate; AlloX pointer repo plus Kubernetes fork and simulator are cloned. Java is present when the java probe passes, but the shipped simulator bin is incomplete on this host and javac is unavailable to rebuild the missing inner-class outputs; the run script also expects experiment arguments. |
| Optimus | true | false | true | false | Optimus scheduler/template files are present; Python2 numpy/jinja2 and Kubernetes/MXNet wiring block executable same-workload smoke. | none for policy/service-unit adapter; direct binary row remains separate; Optimus public runtime is cloned, but current-host smoke is blocked by Python2 dependencies such as numpy and jinja2; full comparison also needs Kubernetes/MXNet wiring. |

## Scope

Finite registered-SOTA policy/service-unit adapter closure.  This is the theorem-facing adapter universe: every registered non-adjacent SOTA family is represented in the measured-cache candidate action union.  It is not direct external-binary superiority for systems without executable same-workload adapter rows.
