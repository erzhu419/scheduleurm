# BAPR Ant Real Dense Service Curve Validation

Date: 2026-06-03

This note records the first full real-RL dense service curve for the BAPR Ant
workload on `jtl110gpu2:GPU1`.  The purpose is to calibrate the service-rate
surface used by the Scheduleurm adaptive scheduling theory and experiments, not
to replace the legacy scheduler policy directly.

## Workload

- Repository: `/home/erzhu419/mine_code/BAPR`
- Python: `/home/erzhu419/miniconda3/envs/bapr-jax/bin/python`
- Command template: `algorithm/experiments/templates/bapr_ant_real.cmd.tpl`
- Environment: `Ant-v2`, `algo=bapr`, `backend=spring`, `env_type=discrete_mode`,
  `mean_dwell_iters=60`, `use_regime_belief`, `penalty_scale=0.5`,
  `critic_target_mode=min`, `no_per_trans_belief`
- Progress unit: BAPR training iteration
- Per-task cap for measurement jobs: `max_iters=30`
- Clean benchmark mode: `--hard-rule-mode clean_bench`
- Pinning: every profile is pinned to `jtl110gpu2:GPU1`
- Warmup: every running task must reach at least iteration 5 before measurement

## Runs

- `module13_bapr_ant_real_dense_jtl110gpu2_gpu1_20260603_001`
  measured 1-6/GPU.  This run used the old last-sample selector, so the report
  below recomputes 1-6 from `events.jsonl` with the new median measurement
  selector.
- `module14_bapr_ant_real_dense_tail_jtl110gpu2_gpu1_20260603_001`
  measured 7-8/GPU with the median selector.
- `module15_bapr_ant_real_9gpu_boundary_jtl110gpu2_gpu1_20260603_001`
  validated the 9/GPU runtime capacity boundary.  The runner now detects
  warmup-time OOM immediately instead of waiting for the warmup timeout.

## Runner Fixes Validated

- Measurement summary selection now uses the median aggregate-rate sample over
  the measurement window, while retaining raw last-sample diagnostics.
- Warmup can require a minimum progress unit, which avoids using compile-only or
  first-eval rates as the service estimate.
- Warmup-time runtime OOM is now treated as a capacity boundary immediately.
- Tail-only service-curve verdicts no longer require 2/GPU and 3/GPU when the
  `min_two_vs_three_gain` check is disabled.

## Service Curve

| Tasks/GPU | Status | Aggregate iter/s | Mean iter/s/task | Selection | Notes |
|---:|:---|---:|---:|:---|:---|
| 1 | valid | 0.222222 | 0.222222 | robust replay | single task |
| 2 | valid | 0.213910 | 0.106955 | robust replay | near plateau |
| 3 | valid | 0.211268 | 0.070423 | robust replay | old last sample was noisy |
| 4 | valid | 0.211922 | 0.052980 | robust replay | near plateau |
| 5 | valid | 0.211686 | 0.042337 | robust replay | near plateau |
| 6 | valid | 0.211890 | 0.035315 | robust replay | old last sample was noisy |
| 7 | valid | 0.200989 | 0.028713 | median | still usable, lower plateau |
| 8 | valid | 0.177566 | 0.022196 | median | clear degradation |
| 9 | boundary | 0.028457 | 0.005691 | OOM boundary | two tasks failed with OOM |

The 9/GPU boundary run recorded:

- `t6588: err_pattern: Traceback (most recent call, Error:, out of memory`
- `t6592: err_pattern: Traceback (most recent call, Error:, out of memory`

## Completion Proxy

For identical tasks with 30 iterations each, the fast-forward makespan proxy is
`ceil(n / k) * 30 / mean_rate(k)`, where `k` is tasks per GPU.

| Tasks/GPU | Makespan proxy, n=120 | Mean-flow proxy, n=120 |
|---:|---:|---:|
| 1 | 16200.0 s | 8167.5 s |
| 2 | 16829.5 s | 8555.0 s |
| 3 | 17039.9 s | 8732.9 s |
| 4 | 16987.5 s | 8776.9 s |
| 5 | 17006.4 s | 8857.5 s |
| 6 | 16989.9 s | 8919.7 s |
| 7 | 18807.0 s | 9481.8 s |
| 8 | 20274.2 s | 10812.9 s |

For this exact BAPR Ant template, 1/GPU has the best measured aggregate service
rate.  Profiles 2-6/GPU are close enough to form a broad plateau, but they do
not improve total makespan under the equal-task proxy.  8/GPU is degraded, and
9/GPU is infeasible on this GPU.

## Scheduling Implication

The admissible profile range for this workload on `jtl110gpu2:GPU1` is 1-8/GPU,
with 9/GPU excluded by a runtime OOM boundary.  For clean throughput experiments,
the adaptive policy should use the calibrated aggregate service curve rather
than a fixed assumption that 4-5/GPU is always optimal.  A conservative initial
candidate set is `{1,2,3,4,5,6,7,8}`, with 9/GPU marked infeasible for this
fabric/workload bucket.

The result still supports the paper route: Scheduleurm can expose finite
co-location actions, measure a bucketed service curve, identify a fabric-specific
capacity boundary, and feed those calibrated rates into the robust candidate-set
MaxWeight layer.

## Remaining Experimental Work

- Repeat the same dense measurement for the user's exact long-running production
  RE-SAC/BAPR commands, because this short Ant template shows aggregate plateau
  behavior but not per-task 4/5/GPU ~= 1/GPU behavior.
- Run multi-GPU/multi-node validation so the service curve is not tied to a
  single 12GB GPU.
- Use longer measurement windows after the first calibration pass to reduce
  eval-cycle noise and tighten `epsilon_est`.
