# Full-Stack Runtime Readiness

| Quantity | Value |
|---|---:|
| `pass` | true |
| `runtime_base_ready` | true |
| `direct_fullstack_sota_superiority_ready` | false |
| `remote` | `huiwei@jtl110gpu` |
| `hostname` | `huiwei-Super-Server` |
| `docker_default_runtime` | `nvidia` |
| `kubernetes_node_ready` | true |
| `nvidia_device_plugin_running` | true |
| `gpu_smoke_completed` | true |

## GPU Capacity

| Field | Value |
|---|---|
| `capacity.nvidia.com/gpu` | `2` |
| `allocatable.nvidia.com/gpu` | `2` |
| `nvidia-smi` | `0, NVIDIA GeForce RTX 3080 Ti, 10 MiB, 12288 MiB / 1, NVIDIA GeForce RTX 3080 Ti, 243 MiB, 12288 MiB` |

## SOTA Runtime Rows

| Adapter | Runtime base | Same-workload runtime | Blocker |
|---|---:|---:|---|
| `pollux_adaptdl_scheduler` | true | true |  |
| `sia_goodput_scheduler` | true | true |  |
| `iadeep_kubernetes_extender` | true | true |  |
| `salus_gpu_sharing` | true | true |  |

## GPU Smoke Log

```text
0, NVIDIA GeForce RTX 3080 Ti, 12288 MiB
CUDA_VISIBLE_DEVICES=unset
NVIDIA_VISIBLE_DEVICES=void
```

## Scope

Certifies the remote Docker/K3s/NVIDIA device-plugin substrate for direct full-stack SOTA experiments.  It does not claim direct full-stack superiority over any SOTA system until that system's own runtime completes the same workload and is compared against Scheduleurm on JCT/makespan.
