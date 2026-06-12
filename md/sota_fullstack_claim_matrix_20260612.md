# SOTA Full-Stack Claim Matrix

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `all_seed_families_present` | true |
| `direct_fullstack_same_workload_ready_count` | 0 |

## System Rows

| Adapter | Smoke | Seeds | Native microbaseline | Direct full-stack ready | Allowed claim | Forbidden claim |
|---|---:|---:|---:|---:|---|---|
| `gavel_simulation` | true | 15 | true | false | native Gavel trace compatibility and bounded native simulator microbaseline; policy-semantics replay remains the cross-system performance comparison | directly outperforms the external system binary or full stack |
| `pollux_adaptdl_scheduler` | true | 5 | false | false | seed readiness and local entrypoint/tool blocker reporting | directly outperforms the external system binary or full stack |
| `iadeep_kubernetes_extender` | false | 5 | false | false | seed readiness and local entrypoint/tool blocker reporting | directly outperforms the external system binary or full stack |
| `salus_gpu_sharing` | false | 5 | false | false | seed readiness and local entrypoint/tool blocker reporting | directly outperforms the external system binary or full stack |
| `decima_simulator` | true | 5 | false | false | Decima seed readiness and entrypoint smoke; scope is Spark-DAG simulator, not GPU co-location baseline | directly outperforms the external system binary or full stack |

## Scope

Claim matrix for external SOTA systems.  It certifies seed readiness and local smoke status, not full-stack superiority unless an adapter row has direct_fullstack_same_workload_ready=true.
