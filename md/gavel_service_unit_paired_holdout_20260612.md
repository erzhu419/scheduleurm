# Gavel Service-Unit Paired Holdout

| Quantity | Value |
|---|---:|
| `pass` | true |
| `row_count` | 12 |
| `usable_row_count` | 12 |
| `policy` | `fifo` |

| Taskset | Split | Jobs | Gavel s | Scheduleurm s | Scale | Usable |
|---|---|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | `calibration` | 1 | 25.464 | 69.5957 | 2.7331 | true |
| `q01_gpu_bound_compute` | `calibration` | 2 | 25.464 | 69.5957 | 2.7331 | true |
| `q01_gpu_bound_compute` | `calibration` | 3 | 25.464 | 139.191 | 5.4662 | true |
| `q01_gpu_bound_compute` | `holdout` | 4 | 25.464 | 139.191 | 5.4662 | true |
| `q01_gpu_bound_compute` | `holdout` | 6 | 50.929 | 208.787 | 4.09957 | true |
| `q01_gpu_bound_compute` | `holdout` | 8 | 50.929 | 273.865 | 5.37739 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 2 | 11.149 | 480 | 43.0532 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 4 | 11.149 | 960 | 86.1064 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 6 | 11.149 | 1434.51 | 128.667 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 8 | 11.149 | 1920 | 172.213 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 12 | 22.297 | 2869.02 | 128.673 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 16 | 22.297 | 3840 | 172.22 | true |

## Scope

Paired bounded-window service-unit calibration between Gavel native simulator makespan and Scheduleurm measured-service replay makespan. This artifact is not a direct full-stack SOTA superiority result.
