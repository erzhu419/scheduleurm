# Salus Remote Full-Stack Attempt

| Quantity | Value |
|---|---:|
| `pass` | `True` |
| `official_images_ready` | `True` |
| `image_pull_blocker_resolved` | `True` |
| `native_runtime_hang_suspected` | `False` |
| `salus_runtime_hang_suspected` | `False` |
| `salus_server_started` | `True` |
| `native_tensorflow_salus_client_completed` | `True` |
| `tensorflow_salus_client_completed` | `True` |
| `same_service_scale_ready` | `True` |
| `scoped_salus_fullstack_same_workload_ready` | `True` |
| `scoped_salus_same_workload_superiority_ready` | `True` |
| `direct_fullstack_sota_superiority_ready` | `False` |
| `native_wall_s` | `199.05483388900757` |
| `salus_wall_s` | `201.74355292320251` |

## Attempts

| Attempt | Completed | Images pulled | Status | Reason |
|---|---:|---:|---|---|
| `native_tensorflow_salus` | true | true | `MARKER_SEEN` | native-client:terminated:Completed:exit=0 |
| `salus_server_client` | true | true | `MARKER_SEEN` | salus-client:terminated:Completed:exit=0; salus-server:running |

## Blockers

| Blocker |
|---|
| none |

## Scope

Scoped remote Salus full-stack attempt on the isolated jtl110gpu K3s substrate.  It does not modify production scheduler code and does not alter system CUDA/cuDNN.  It counts Salus only when a TensorFlow-Salus client completes through a Salus server.
