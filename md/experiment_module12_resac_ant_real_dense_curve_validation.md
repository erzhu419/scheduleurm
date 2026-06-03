# Module 12: True RE-SAC Ant Dense Service-Curve Validation

Date: 2026-06-03

Primary runs:

- `module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_002`
- `module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_003`

Run directories:

- `/home/erzhu419/.claude/scheduler/experiments/runs/module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_002`
- `/home/erzhu419/.claude/scheduler/experiments/runs/module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_003`

Workload: true RE-SAC `resac` on `Ant-v2`, `ensemble_size=10`, `max_iters=80`, stationary mode, one pinned GPU profile at a time on `jtl110gpu2:GPU1`.

## Why This Replaces The Smoke Curve

The earlier dense curve used a smoke RE-SAC/SAC template and selected low co-location for makespan. That was not enough for the user's real workload claim that 4-5 concurrent RL tasks can have near-1/GPU single-task speed.

This run uses the real RE-SAC Ant command template from `algorithm/experiments/templates/resac_ant_real.cmd.tpl`. It confirms that the realistic curve is flatter in aggregate: valid 10/GPU has the highest observed aggregate service rate, 12/GPU is close, and 13/GPU is the first true runtime capacity boundary.

## Runner And Logging Changes

- `status --json --brief --ids ...` now scopes experiment status refreshes to the tasks under measurement, so raw logs do not include the entire queue.
- Workload service-curve validation now detects runtime capacity boundaries after launch, not only scheduler reservation boundaries. CUDA/JAX OOM and resource-exhausted patterns are recorded as boundary reasons and stop higher profiles when `--stop-on-capacity-boundary` is enabled.
- The JAX runtime patch in `algorithm/runtime_patches/sitecustomize.py` is used only as an experiment environment patch for conda-forge CPython version strings. It does not change scheduler policy.

## Dense Curve

Profiles 1-8 used `vram_mb=1400`. Profile 9 initially hit scheduler reservation rather than real workload capacity, so the corrected 9-13 tail was rerun with `vram_mb=820` and `cpu=0`. The 13/GPU profile launched but one task failed with runtime OOM; that is treated as the capacity boundary. The partial 14/GPU attempt after the OOM boundary is intentionally not used.

| Count/GPU | Running | With rate | Placement | Boundary | Mean iter/s | Aggregate iter/s | Min iter/s | Max iter/s | Blocks | Note |
|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|:---|
| 1 | 1 | 1 | ok | no | 0.344828 | 0.344828 | 0.344828 | 0.344828 | 0 |  |
| 2 | 2 | 2 | ok | no | 0.166667 | 0.333333 | 0.166667 | 0.166667 | 0 |  |
| 3 | 3 | 3 | ok | no | 0.111536 | 0.334609 | 0.109890 | 0.112360 | 0 |  |
| 4 | 4 | 4 | ok | no | 0.072231 | 0.288922 | 0.038911 | 0.084034 | 0 |  |
| 5 | 5 | 5 | ok | no | 0.057659 | 0.288295 | 0.005559 | 0.081967 | 0 |  |
| 6 | 6 | 6 | ok | no | 0.050948 | 0.305689 | 0.026042 | 0.056180 | 0 |  |
| 7 | 7 | 7 | ok | no | 0.043232 | 0.302622 | 0.024155 | 0.051546 | 0 |  |
| 8 | 8 | 8 | ok | no | 0.042301 | 0.338410 | 0.006203 | 0.063694 | 0 |  |
| 9 | 9 | 9 | ok | no | 0.035385 | 0.318461 | 0.007199 | 0.062500 | 0 |  |
| 10 | 10 | 10 | ok | no | 0.035617 | 0.356167 | 0.009747 | 0.073529 | 0 |  |
| 11 | 11 | 11 | ok | no | 0.029217 | 0.321386 | 0.005195 | 0.060606 | 0 |  |
| 12 | 12 | 12 | ok | no | 0.028901 | 0.346811 | 0.005750 | 0.070922 | 0 |  |
| 13 | 12 | 12 | invalid | yes | 0.023055 | 0.276658 | 0.007348 | 0.072464 | 1 | OOM boundary |

Boundary reason:

`t6484: err_pattern: Traceback (most recent call, Error:, out of memory`

## Completion Proxies

For total-task completion, the relevant proxy is makespan, not single-task speed. For `n` total tasks and profile count `k`, the runner uses:

`makespan_proxy_s = ceil(n / running_slots) * 80 / mean_iter_per_s`

The mean-flow proxy favors low concurrency because early waves complete sooner; the makespan proxy is the better check for "finish all n tasks".

### n=12

Best mean-flow count: 1/GPU. Best makespan count: 12/GPU.

| Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|
| 1 | 1508.000 | 2784.000 |
| 2 | 1680.000 | 2880.000 |
| 3 | 1793.137 | 2869.018 |
| 4 | 2215.130 | 3322.695 |
| 5 | 2428.065 | 4162.397 |
| 6 | 2355.338 | 3140.451 |
| 7 | 2621.533 | 3700.987 |
| 8 | 2521.599 | 3782.399 |
| 9 | 2826.094 | 4521.750 |
| 10 | 2620.493 | 4492.273 |
| 11 | 2966.320 | 5476.284 |
| 12 | 2768.082 | 2768.082 |

### n=24

Best mean-flow count: 1/GPU. Best makespan count: 12/GPU.

| Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|
| 1 | 2900.000 | 5568.000 |
| 2 | 3120.000 | 5760.000 |
| 3 | 3227.646 | 5738.037 |
| 4 | 3876.478 | 6645.390 |
| 5 | 4046.775 | 6937.328 |
| 6 | 3925.563 | 6280.901 |
| 7 | 4163.611 | 7401.974 |
| 8 | 3782.399 | 5673.598 |
| 9 | 4239.141 | 6782.625 |
| 10 | 3930.739 | 6738.410 |
| 11 | 4449.480 | 8214.426 |
| 12 | 4152.123 | 5536.164 |

### n=60

Best mean-flow count: 1/GPU. Best makespan count: 10/GPU.

| Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|
| 1 | 7076.000 | 13920.000 |
| 2 | 7440.000 | 14400.000 |
| 3 | 7531.173 | 14345.092 |
| 4 | 8860.520 | 16613.476 |
| 5 | 9018.527 | 16649.588 |
| 6 | 8636.239 | 15702.253 |
| 7 | 8882.369 | 16654.443 |
| 8 | 8069.117 | 15129.595 |
| 9 | 8704.369 | 15826.126 |
| 10 | 7861.478 | 13476.820 |
| 11 | 8898.961 | 16428.851 |
| 12 | 8304.245 | 13840.409 |

### n=120

Best mean-flow count: 1/GPU. Best makespan count: 10/GPU.

| Count/GPU | Mean flow proxy s | Makespan proxy s |
|---:|---:|---:|
| 1 | 14036.000 | 27840.000 |
| 2 | 14640.000 | 28800.000 |
| 3 | 14703.720 | 28690.185 |
| 4 | 17167.258 | 33226.952 |
| 5 | 17343.321 | 33299.176 |
| 6 | 16487.366 | 31404.506 |
| 7 | 16793.230 | 33308.885 |
| 8 | 15129.595 | 28367.990 |
| 9 | 16221.779 | 31652.251 |
| 10 | 14599.888 | 26953.640 |
| 11 | 16314.762 | 30119.560 |
| 12 | 15224.450 | 27680.818 |

## Interpretation

This run supports the paper route in `math.md`: actions must include concrete task-to-GPU co-location choices, and service calibration must use workload-specific aggregate service curves. For this true RE-SAC Ant workload, the high co-location region is not a corner case. It is part of the measured feasible service surface.

The result also explains the apparent contradiction with single-task speed. If the objective is the completion time of one task, 1/GPU looks best. If the objective is finishing a batch of many homogeneous RE-SAC jobs, 10/GPU or 12/GPU can finish the whole batch sooner because aggregate service remains high before the 13/GPU OOM boundary.

## Validation

- `python3 -m py_compile skill/scheduler.py algorithm/experiments/sweetspot_ab_validation.py algorithm/experiments/service_curve_validation.py algorithm/experiments/workload_service_curve_validation.py algorithm/runtime_patches/sitecustomize.py`
- Targeted workload service-curve regression: 12 checks passed, 0 failed.
- Merged verdict over valid 1-12 plus 13/GPU OOM boundary: pass, no failure reasons.

## Next Module

The next module should run the same dense protocol on BAPR/RE-SAC variants that the user actually queues in production. Keep the same rule: profile one module, validate performance and logging, push, then move to the next module.
