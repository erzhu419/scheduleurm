# Named External Runtime Probe Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `NAMED_SAME_HOST_RUNTIME_PROBE_PASS_DIRECT_SOTA_FALSE` |
| `gate_pass` | true |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `pass_meaning` | named five same-host same-workload runtime probes with paired native-better rows are closed; direct full-stack SOTA superiority, registered-universe external-binary superiority, original multi-node superiority, and production-wide trace superiority are not claimed |
| `named_same_host_runtime_probe_ready` | true |
| `named_same_host_runtime_probe_native_better` | true |
| `direct_fullstack_sota_superiority_ready` | false |
| `direct_fullstack_named_sota_superiority_ready` | false |
| `registered_sota_universe_superiority_ready` | false |
| `arbitrary_sota_superiority_ready` | false |
| `multinode_original_deployment_superiority_ready` | false |
| `production_wide_organic_trace_superiority_ready` | false |
| `full_stack_ready_count` | 5 |
| `named_same_host_runtime_probe_ready_count` | 5 |
| `named_same_host_runtime_probe_native_better_count` | 5 |
| `policy_semantics_comparison_ready` | true |
| `hard_blocker_certificate_ready` | true |
| `native_execution_ready_count` | 3 |
| `native_attempt_direct_full_stack_ready_count` | 0 |
| `gavel_same_workload_native_simulator_ready` | true |
| `gavel_physical_same_workload_fullstack_ready` | true |
| `gavel_physical_scoped_same_workload_superiority_ready` | true |
| `pollux_same_workload_fullstack_ready` | true |
| `pollux_scoped_same_workload_superiority_ready` | true |
| `sia_same_workload_fullstack_ready` | true |
| `sia_scoped_same_workload_superiority_ready` | true |
| `iadeep_same_workload_fullstack_ready` | true |
| `iadeep_scoped_same_workload_superiority_ready` | true |
| `salus_same_workload_fullstack_ready` | true |
| `salus_scoped_same_workload_superiority_ready` | true |

## Tool Blockers

| Tool | Available | Path / version |
|---|---:|---|
| `docker` | false | `` |
| `go` | false | `` |
| `kubectl` | false | `` |
| `nvidia-smi` | true | `/usr/lib/wsl/lib/nvidia-smi` |
| `python3` | true | `/usr/bin/python3` |

## System Rows

| Adapter | Smoke | Native microbaseline | Native simulator comparison | Gavel physical pair | Resident-delay JCT holdout | Pollux runtime pair | Sia runtime pair | IADeep runtime pair | Salus runtime pair | Runtime probe ready | Native-better probe | Reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gavel_simulation` | true | true | true | true | true | false | false | false | false | true | true | Gavel completed the same temporary CUDA probe through its physical scheduler/worker/RPC/dispatcher/GavelIterator path, and the same script run directly on the same host has lower JCT in the scoped run; this is a Gavel-scoped physical full-stack row, not an all-workload Gavel claim |
| `pollux_adaptdl_scheduler` | true | false | false | false | false | true | false | false | false | true | true | Pollux/AdaptDL completed the same CUDA binary through its Kubernetes controller/allocator/supervisor stack, and the native Scheduleurm-controlled Docker path has lower JCT on the scoped short and long probe cases; this is a Pollux-scoped full-stack row, not an all-SOTA superiority certificate |
| `sia_goodput_scheduler` | false | false | false | false | false | false | true | false | false | true | true | Sia/AdaptDL MIP completed the same CUDA binary through its Kubernetes controller/allocator path, and the native Scheduleurm-controlled Docker path has lower JCT on the scoped short and long probe cases; this is a Sia-scoped full-stack row, not an all-SOTA superiority certificate |
| `iadeep_kubernetes_extender` | false | false | false | false | false | false | false | true | false | true | true | IADeep completed the same CUDA binary through its Kubernetes scheduler-extender/device-plugin GPU-sharing path, and the native Scheduleurm-controlled Docker path has lower JCT on the scoped stable probe case; this is an IADeep-scoped full-stack row, not an all-SOTA superiority certificate |
| `salus_gpu_sharing` | false | false | false | false | false | false | false | false | true | true | true | Salus completed the same TensorFlow-Salus workload through its server/zrpc path, and the native TensorFlow-Salus path on the same host has lower wall time in the scoped run; this is a Salus-scoped full-stack row, not an arbitrary future-workload claim |
| `decima_simulator` | true | false | false | false | false | false | false | false | false | false | false | Decima is a Spark-DAG simulator, so even a runnable entrypoint is not a GPU co-location system baseline |

## Scope

Scoped named-system same-host runtime-probe evidence for Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus.  The gate does not claim direct full-stack SOTA superiority, arbitrary SOTA superiority, future-workload superiority, original multi-node deployment superiority, or production-wide online trace superiority.
