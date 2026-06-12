# Fabric-Cover Metric Contract Certificate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `exact_measured_population_ready` | true |
| `general_future_population_ready` | false |

## Feature Map

| Component | Definition |
|---|---|
| finite keys | `class_key, regime_key, post_count_bucket, post_vram_bucket, util_bucket, resource_bucket, task_kind` |
| numeric scales | `{"legacy_runtime_s": 3600.0, "post_task_count": 8.0, "post_vram_frac": 1.0, "running_task_count": 8.0, "used_vram_frac": 1.0, "util_pct": 100.0}` |
| metric | Hamming distance over finite keys plus clipped scaled l1 distance over numeric features, as implemented by algorithm.features.finite_feature_metric |

## Taskset Contract Rows

| Taskset | Actions | L | exact rho | exact Lrho | best compressed k | best compressed rho | best compressed Lrho | Exact usable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q00_light_control` | 13 | 113.440320600 | 0.000000000 | 0.000000000 | 8 | 3.250000000 | 368.681041950 | true |
| `q01_gpu_bound_compute` | 8 | 1.629437538 | 0.000000000 | 0.000000000 | 4 | 3.250000000 | 5.295672000 | true |
| `q10_cpu_host_bound` | 9 | 10.083779263 | 0.000000000 | 0.000000000 | 4 | 3.250000000 | 32.772282605 | true |
| `q11_cpu_gpu_coupled` | 9 | 0.024699809 | 0.000000000 | 0.000000000 | 4 | 3.250000000 | 0.080274379 | true |
| `hybrid_research_portfolio` | 243 | 10.083779263 | 0.000000000 | 0.000000000 | 16 | 2.000000000 | 20.167558526 | true |

## Exclusions

- future unseen node/GPU regimes outside the measured finite action population
- unprofiled topology jumps where the finite metric has no sensitivity sample
- candidate generators claiming smaller covers without spending the reported Lrho
- statistical envelopes over future perturbation samples without a failure-probability table

## Scope

Submission-facing metric contract for the measured finite fabric-cover claim. It closes exact measured-slice rho=0 and reports compressed-cover Lrho. It intentionally does not certify all future feasible cluster states.
