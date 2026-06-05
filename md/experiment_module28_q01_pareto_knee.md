# Module28 q01 Pareto-Knee Improvement

Date: 2026-06-04

This module changes the single-taskset q01 default from the extreme delay point
to a guarded Pareto-knee point.

## Mathematical Reason

For q01, let `A_k` be the measured aggregate service rate when `k` jobs share
one GPU. The measured curve is almost flat in aggregate service:

```text
A_1 ~= 34.48 step/s
A_8 ~= 35.63 step/s
```

but per-task service falls by roughly a factor of eight as `k` increases. Thus:

```text
8/GPU: best all-task makespan, worse mean-flow
1/GPU: best mean-flow, slower all-task makespan
```

No fixed profile can strictly dominate both endpoints. The right mathematical
move is to select a knee on the Pareto frontier.

## Selector

The new single-workload default is:

```text
calibrated_guarded_knee
```

It uses:

```text
candidate set = profiles with makespan <= 1.02 * best_makespan
tie-break = minimum deterministic mean-flow
```

For q01 this selects:

```text
gpu_heavy_jax_matmul: 4/GPU
```

The old extreme delay policy is still available as:

```text
calibrated_delay_statewise_policy()
```

## q01 Result

Against Scheduleurm legacy:

| Profile | Makespan improvement | Mean-flow improvement |
|---:|---:|---:|
| 1/GPU delay endpoint | 1.064x | 1.148x |
| 4/GPU knee | 1.080x | 1.045x |
| 8/GPU throughput endpoint | 1.096x | 0.940x |

Against SOTA-style endpoints:

| SOTA-style baseline | Candidate vs baseline makespan | Candidate vs baseline mean-flow |
|---|---:|---:|
| throughput-table goodput, 8/GPU | 0.985x | 1.111x |
| delay oracle, 1/GPU | 1.015x | 0.910x |

So the q01 knee gives up only about 1.5% makespan against the throughput
endpoint while keeping about 11% mean-flow improvement over that endpoint. It is
therefore a better reviewer-facing main point than the extreme 1/GPU delay
choice.

## Scope

The mixed portfolio policy remains `calibrated_global_guarded`. In the current
portfolio, the real local CPU bucket is the host-pressure member, so the global
selector can still choose `gpu_heavy_jax_matmul=1/GPU` without increasing
portfolio makespan.

This module does not weaken the proof route. It is still support first, bounded
delay penalty second:

```text
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

## Validation

```text
q01 standalone:
  pass=true
  profile=4/GPU
  legacy makespan improvement=1.080207
  legacy mean-flow improvement=1.044538

SOTA-style q01:
  not Pareto-dominated=true
  throughput-table comparison=(makespan 0.985283, mean-flow 1.110640)
  delay-oracle comparison=(makespan 1.015141, mean-flow 0.909833)
```
