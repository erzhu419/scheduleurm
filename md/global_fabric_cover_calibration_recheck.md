# Global Fabric-Cover Perturbation Calibration

## Summary

| Taskset | Actions | L | exact rho | exact Lrho | representative rho | representative Lrho | Main usable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | 13 | 113.440320600 | 0.000000000 | 0.000000000 | 5.000000000 | 567.201603000 | true |
| `q01_gpu_bound_compute` | 8 | 1.629437538 | 0.000000000 | 0.000000000 | 4.000000000 | 6.517750154 | true |
| `q10_cpu_host_bound` | 9 | 10.083779263 | 0.000000000 | 0.000000000 | 4.750000000 | 47.897951500 | true |
| `q11_cpu_gpu_coupled` | 9 | 0.024699809 | 0.000000000 | 0.000000000 | 4.750000000 | 0.117324092 | true |
| `hybrid_research_portfolio` | 243 | 10.083779263 | 0.000000000 | 0.000000000 | 4.750000000 | 47.897951500 | true |

## Scope

calibrates the finite-feature metric and service Lipschitz envelope on measured Scheduleurm tasksets. The main theorem certificate uses the exact measured finite action slice with rho=0; representative cover radii are reported as extension/engineering diagnostics.

## Perturbation Witnesses

### `q00_light_control`

- worst class: `light_control_local`
- profiles: `1` to `13`
- distance: `5.000000000`
- service gap: `567.201603000`
- ratio: `113.440320600`

### `q01_gpu_bound_compute`

- worst class: `gpu_heavy_jax_matmul`
- profiles: `3` to `4`
- distance: `3.250000000`
- service gap: `5.295672000`
- ratio: `1.629437538`

### `q10_cpu_host_bound`

- worst class: `cpu_heavy_local_bench`
- profiles: `1` to `8`
- distance: `4.750000000`
- service gap: `47.897951500`
- ratio: `10.083779263`

### `q11_cpu_gpu_coupled`

- worst class: `hybrid_rl_resac_ant`
- profiles: `1` to `2`
- distance: `3.250000000`
- service gap: `0.080274379`
- ratio: `0.024699809`

### `hybrid_research_portfolio`

- worst class: `cpu_heavy_local_bench`
- profiles: `1` to `8`
- distance: `4.750000000`
- service gap: `47.897951500`
- ratio: `10.083779263`
