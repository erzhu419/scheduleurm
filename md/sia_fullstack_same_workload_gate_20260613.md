# Sia Full-Stack Same-Workload Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `sia_fullstack_completed` | true |
| `native_pair_completed` | true |
| `same_service_scale_ready` | true |
| `scoped_sia_same_workload_superiority_ready` | true |
| `direct_fullstack_sota_superiority_ready` | false |

## Paired Cases

| Case | Sia phase | Sia CRD JCT (s) | Native wall (s) | Native/Sia JCT | Sia steps/s | Native steps/s | Native/Sia service |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda_probe_short` | Succeeded | 36.946657 | 1.425378 | 0.038579 | 302.018045 | 302.014335 | 0.999988 |
| `cuda_probe_long` | Succeeded | 61.363175 | 10.308755 | 0.167996 | 332.249588 | 332.768032 | 1.001560 |

## Compatibility Fixes

| Component | Fix | Semantics changed | Reason |
|---|---|---:|---|
| Sia/AdaptDL scheduler image | pin scheduler runtime dependencies to NumPy < 1.24 and pandas < 2.0 | false | upstream Sia/AdaptDL artifact uses deprecated NumPy/Pandas APIs |
| Sia/AdaptDL supervisor | cast ADAPTDL_SUPERVISOR_SERVICE_PORT to int | false | Kubernetes service environment variables are strings and aiohttp requires an integer port |
| Sia physical-cluster mapping | map the K3s node name huiwei-super-server to the existing rtx cluster type | false | the artifact assumes a hard-coded Phoebe cluster node inventory |
| Sia MIP policy adapter | lift a scalar single-cluster speedup function into the artifact's {gpu_type: fn} map | false | single-GPU-type Kubernetes deployments do not need a multi-cluster speedup dictionary |

## Scope

Scoped same-host same-binary full-stack evidence for the Sia physical-cluster AdaptDL/MIP artifact.  It supports a claim that the Sia runtime can be run and compared on this cluster, and that the native Scheduleurm-controlled Docker path has lower JCT on these two CUDA probe cases.  It does not prove direct full-stack superiority over Gavel, Pollux, IADeep, Salus, or arbitrary future workloads.
