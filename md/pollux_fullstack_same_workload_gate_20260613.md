# Pollux Full-Stack Same-Workload Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `pollux_fullstack_completed` | true |
| `native_pair_completed` | true |
| `same_service_scale_ready` | true |
| `scoped_pollux_same_workload_superiority_ready` | true |
| `direct_fullstack_sota_superiority_ready` | false |

## Paired Cases

| Case | Pollux phase | Pollux CRD JCT (s) | Native wall (s) | Native/Pollux JCT | Pollux steps/s | Native steps/s | Native/Pollux service |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda_probe_short` | Succeeded | 3.934861 | 1.425378 | 0.362243 | 302.027281 | 302.014335 | 0.999957 |
| `cuda_probe_long` | Succeeded | 13.287994 | 10.308755 | 0.775795 | 301.795277 | 332.768032 | 1.102628 |

## Compatibility Fixes

| Component | Fix | Semantics changed | Reason |
|---|---|---:|---|
| Pollux/AdaptDL scheduler image | pin sched NumPy dependency to numpy>=1.17,<1.24 | false | upstream Pollux code uses np.int, removed by NumPy 1.24 |
| Pollux/AdaptDL supervisor | cast ADAPTDL_SUPERVISOR_SERVICE_PORT to int | false | Kubernetes service environment variables are strings and aiohttp requires an integer port |

## Scope

Scoped same-host same-binary full-stack evidence for Pollux/AdaptDL. It supports a claim that the Pollux/AdaptDL runtime can be run and compared on this cluster, and that the native Scheduleurm-controlled Docker path has lower JCT on these two CUDA probe cases.  It does not by itself prove direct full-stack superiority over Gavel, Sia, IADeep, Salus, or arbitrary future workloads.
