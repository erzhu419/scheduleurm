# Gavel Service-Unit Paired Holdout

| Quantity | Value |
|---|---:|
| `pass` | true |
| `row_count` | 12 |
| `usable_row_count` | 12 |
| `policy` | `fifo` |

| Taskset | Split | Jobs | Gavel s | Scheduleurm s | Scale | Usable |
|---|---|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | `calibration` | 1 | 25.464 | 51.2704 | 2.01345 | true |
| `q01_gpu_bound_compute` | `calibration` | 2 | 25.464 | 51.2704 | 2.01345 | true |
| `q01_gpu_bound_compute` | `calibration` | 3 | 25.464 | 102.541 | 4.02689 | true |
| `q01_gpu_bound_compute` | `holdout` | 4 | 25.464 | 102.541 | 4.02689 | true |
| `q01_gpu_bound_compute` | `holdout` | 6 | 50.929 | 153.811 | 3.02011 | true |
| `q01_gpu_bound_compute` | `holdout` | 8 | 50.929 | 205.082 | 4.02681 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 2 | 11.149 | 406.206 | 36.4343 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 4 | 11.149 | 812.413 | 72.8686 | true |
| `q11_cpu_gpu_coupled` | `calibration` | 6 | 11.149 | 1218.62 | 109.303 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 8 | 11.149 | 1624.83 | 145.737 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 12 | 22.297 | 2437.24 | 109.308 | true |
| `q11_cpu_gpu_coupled` | `holdout` | 16 | 22.297 | 3249.65 | 145.744 | true |

## Scope

Paired bounded-window service-unit calibration between Gavel native simulator makespan and Scheduleurm measured-service replay makespan. This artifact is not a direct full-stack SOTA superiority result.
