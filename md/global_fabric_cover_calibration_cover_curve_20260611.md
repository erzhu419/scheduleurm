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

calibrates the finite-feature metric and service Lipschitz envelope on measured Scheduleurm tasksets. The main theorem certificate uses the exact measured finite action slice with rho=0. Greedy k-center cover curves over the same finite feature metric report how much candidate-cover slack Lrho would be consumed when the candidate family is smaller than the measured full action slice.

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

## Greedy Candidate-Cover Curve

Each row is computed on the same finite-feature metric used by the
fabric-cover theorem.  The exact measured finite-slice certificate is
the final row with rho=0; earlier rows quantify the support slack that
a smaller candidate generator would spend.

| Taskset | Candidate count | rho | Lrho | Worst uncovered action |
|---|---:|---:|---:|---|
| `q00_light_control` | 1 | 5.000000000 | 567.201603000 | `profile_combo:light_control_local=1` |
| `q00_light_control` | 2 | 4.500000000 | 510.481442700 | `profile_combo:light_control_local=7` |
| `q00_light_control` | 4 | 3.750000000 | 425.401202250 | `profile_combo:light_control_local=10` |
| `q00_light_control` | 8 | 3.250000000 | 368.681041950 | `profile_combo:light_control_local=6` |
| `q00_light_control` | 13 | 0.000000000 | 0.000000000 | `` |
| `q01_gpu_bound_compute` | 1 | 4.000000000 | 6.517750154 | `profile_combo:gpu_heavy_jax_matmul=8` |
| `q01_gpu_bound_compute` | 2 | 3.750000000 | 6.110390769 | `profile_combo:gpu_heavy_jax_matmul=1` |
| `q01_gpu_bound_compute` | 4 | 3.250000000 | 5.295672000 | `profile_combo:gpu_heavy_jax_matmul=2` |
| `q01_gpu_bound_compute` | 8 | 0.000000000 | 0.000000000 | `` |
| `q10_cpu_host_bound` | 1 | 4.750000000 | 47.897951500 | `profile_combo:cpu_heavy_local_bench=1` |
| `q10_cpu_host_bound` | 2 | 3.750000000 | 37.814172237 | `profile_combo:cpu_heavy_local_bench=4` |
| `q10_cpu_host_bound` | 4 | 3.250000000 | 32.772282605 | `profile_combo:cpu_heavy_local_bench=2` |
| `q10_cpu_host_bound` | 8 | 3.250000000 | 32.772282605 | `profile_combo:cpu_heavy_local_bench=9` |
| `q10_cpu_host_bound` | 9 | 0.000000000 | 0.000000000 | `` |
| `q11_cpu_gpu_coupled` | 1 | 4.750000000 | 0.117324092 | `profile_combo:hybrid_rl_resac_ant=9` |
| `q11_cpu_gpu_coupled` | 2 | 3.750000000 | 0.092624283 | `profile_combo:hybrid_rl_resac_ant=5` |
| `q11_cpu_gpu_coupled` | 4 | 3.250000000 | 0.080274379 | `profile_combo:hybrid_rl_resac_ant=1` |
| `q11_cpu_gpu_coupled` | 8 | 3.250000000 | 0.080274379 | `profile_combo:hybrid_rl_resac_ant=8` |
| `q11_cpu_gpu_coupled` | 9 | 0.000000000 | 0.000000000 | `` |
| `hybrid_research_portfolio` | 1 | 4.750000000 | 47.897951500 | `profile_combo:cpu_heavy_local_bench=1,gpu_heavy_jax_matmul=1,hybrid_rl_resac_ant=1` |
| `hybrid_research_portfolio` | 2 | 3.750000000 | 37.814172237 | `profile_combo:cpu_heavy_local_bench=4,gpu_heavy_jax_matmul=1,hybrid_rl_resac_ant=1` |
| `hybrid_research_portfolio` | 4 | 3.250000000 | 32.772282605 | `profile_combo:cpu_heavy_local_bench=2,gpu_heavy_jax_matmul=1,hybrid_rl_resac_ant=1` |
| `hybrid_research_portfolio` | 8 | 3.250000000 | 32.772282605 | `profile_combo:cpu_heavy_local_bench=9,gpu_heavy_jax_matmul=1,hybrid_rl_resac_ant=1` |
| `hybrid_research_portfolio` | 16 | 2.000000000 | 20.167558526 | `profile_combo:cpu_heavy_local_bench=7,gpu_heavy_jax_matmul=2,hybrid_rl_resac_ant=1` |
| `hybrid_research_portfolio` | 32 | 2.000000000 | 20.167558526 | `profile_combo:cpu_heavy_local_bench=5,gpu_heavy_jax_matmul=1,hybrid_rl_resac_ant=2` |
| `hybrid_research_portfolio` | 64 | 2.000000000 | 20.167558526 | `profile_combo:cpu_heavy_local_bench=2,gpu_heavy_jax_matmul=2,hybrid_rl_resac_ant=3` |
| `hybrid_research_portfolio` | 128 | 2.000000000 | 20.167558526 | `profile_combo:cpu_heavy_local_bench=3,gpu_heavy_jax_matmul=3,hybrid_rl_resac_ant=5` |
| `hybrid_research_portfolio` | 243 | 0.000000000 | 0.000000000 | `` |
