# Salus Full-Stack Same-Workload Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `official_manifest_ready` | true |
| `docker_image_pull_completed` | false |
| `k8s_image_pull_pod_scheduled` | true |
| `k8s_image_pull_completed` | false |
| `k8s_image_pull_in_progress` | true |
| `k8s_pull_pod_removed_after_timeout` | true |
| `salus_server_help_smoke_completed` | false |
| `remote_fullstack_attempt_ready` | true |
| `remote_fullstack_superiority_ready` | true |
| `remote_image_pull_blocker_resolved` | true |
| `remote_runtime_hang_suspected` | false |
| `official_image_pull_completed` | true |
| `salus_server_started` | true |
| `tensorflow_salus_client_completed` | true |
| `same_service_scale_ready` | true |
| `scoped_salus_fullstack_same_workload_ready` | true |
| `direct_fullstack_sota_superiority_ready` | false |
| `official_manifest_total_layer_bytes` | 928565358 |
| `pull_exit_code` | 124 |
| `pull_elapsed_s` | 180.000 |
| `k8s_image_pull_reason` | `ContainerCreating` |
| `source_build_valid_fullstack_substitute_ready` | false |
| `remote_native_wall_s` | `199.05483388900757` |
| `remote_salus_wall_s` | `201.74355292320251` |

## Runtime Contract

| Field | Value |
|---|---|
| `server_contract` | official Salus server image must start and listen on port 5501 |
| `client_contract` | TensorFlow-Salus workload must create a Session with zrpc://tcp://HOST:5501 and complete |
| `why_cuda_probe_is_insufficient` | plain CUDA binaries bypass Salus and therefore are not a Salus full-stack workload |
| `k8s_image_pull_contract` | K3s/containerd image availability is accepted as a server-image readiness path only when the Salus container starts and the smoke command runs |
| `source_build_contract` | source-build substitution is accepted only with TensorFlow-Salus and Salus' pinned native dependency contract |
| `upstream_runtime_from_dockerfile` | FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04 as prod |
| `upstream_readme_contract_seen` | True |

## Blockers

| Blocker |
|---|
| none |

## Nonblocking Notes

| Note |
|---|
| remote local-mirror path resolved the official Salus and TensorFlow-Salus image availability blocker; both images were loaded into the remote Docker store and used by K3s pods |
| source-build substitution remains unavailable but is not needed for the completed official-image full-stack row |
| upstream Salus production image is pinned to CUDA 9.1/cuDNN7, but the official image path completed the scoped TensorFlow-Salus workload on the current host |
| Salus requires a TensorFlow-Salus client workload; a plain CUDA probe is not a valid Salus full-stack workload |

## Scope

Strict Salus full-stack gate.  It does not count CUDA probes, policy-semantics replay, or server-only startup as full-stack Salus evidence.  Salus becomes comparable only after the official server image and a TensorFlow-Salus client workload complete on the same host with a service-scale bridge.
