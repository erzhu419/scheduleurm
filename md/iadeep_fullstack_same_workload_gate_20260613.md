# IADeep Full-Stack Same-Workload Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `regeneration_mode` | `precomputed_live_run` |
| `current_safe_reproduction` | false |
| `requires_external_stack` | true |
| `current_environment_has_tools` | false |
| `iadeep_fullstack_completed` | true |
| `native_pair_completed` | true |
| `same_service_scale_ready` | true |
| `scoped_iadeep_same_workload_superiority_ready` | true |
| `direct_fullstack_sota_superiority_ready` | false |

## Paired Cases

| Case | IADeep phase | IADeep CRD JCT (s) | Native wall (s) | Native/IADeep JCT | IADeep steps/s | Native steps/s | Native/IADeep service | GPU-sharing assigned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cuda_probe_short_stable` | Succeeded | 5.000000 | 1.425378 | 0.285076 | 302.024196 | 302.014335 | 0.999967 | true |

## Compatibility Fixes

| Component | Fix | Semantics changed | Reason |
|---|---|---:|---|
| IADeep scheduler-extender image | use a reachable Go proxy during image build | false | the upstream Go module download path timed out from the test host |
| IADeep scheduler-extender etcd client | allow HTTP etcd transport for the isolated single-node K3s test etcd | false | the artifact assumes a preconfigured etcd certificate path that is absent in the local test substrate |
| IADeep device-plugin image | set NVIDIA_VISIBLE_DEVICES=all for the local NVIDIA CDI runtime | false | the local container runtime rejects uppercase ALL as a CDI device identifier |
| IADeep independent kube-scheduler deployment | move the scheduler secure port away from the default scheduler hostNetwork port | false | the direct IADeep scheduler runs beside, not instead of, the production K3s scheduler |

## Scope

Scoped same-host same-binary full-stack evidence for IADeep's Kubernetes scheduler-extender/device-plugin GPU-sharing path.  The completed pod carries IADeep GPU-memory assignment annotations and the same CUDA probe binary completed with service scale matching the native Scheduleurm-controlled Docker pair.  This is an IADeep-scoped full-stack row, not an all-SOTA superiority certificate.
