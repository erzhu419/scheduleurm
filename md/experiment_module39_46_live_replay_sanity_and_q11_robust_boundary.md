# Modules39-46 Live Replay Sanity And Q11 Robust Boundary

Date: 2026-06-08

This module batch closes the first two live replay sanity obligations and fixes
one important q11 calibration bug. The result is not a weaker theory claim: it
is a stricter feasible-action certificate. Once a profile produces a runtime
capacity boundary, replay and SOTA-style policies must not keep using that
profile or any higher co-location profile as if it were feasible.

## Q01 Fresh Live Sanity

Run:

```text
module39_q01_live_sanity_jtl110gpu_profile4_20260608_001
```

Workload: `gpu_heavy_jax_matmul`, 48-task q01 trace, candidate profile 4/GPU,
two GPUs on `jtl110gpu`.

Result:

| Metric | Replay predicted | Fresh live proxy | Relative error |
|---|---:|---:|---:|
| makespan_s | 1661.313 | 1652.248 | 0.005457 |
| mean_flow_s | 963.498 | 963.811 | 0.000325 |
| p90_flow_s | 1613.411 | 1652.248 | 0.024072 |

Validation artifact:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module39_q01_live_sanity_jtl110gpu_profile4_20260608_001/reports/q01_live_validation.json
```

`usable_for_live_sanity = true` at 10% relative-error tolerance.

## Q11 Profile-10 Boundary

Profile 10/GPU was previously treated as a valid q11 replay action from module12.
Two fresh live sanity attempts showed that this is not robust enough for the
current theorem-facing feasible family:

| Run | Node/GPU | Profile | Outcome |
|---|---|---:|---|
| `module41_q11_live_sanity_jtl110gpu_profile10_20260608_001` | `jtl110gpu:GPU0` | 10 | runtime OOM, 9/10 running |
| `module42_q11_live_sanity_jtl110gpu2_gpu1_profile10_20260608_001` | `jtl110gpu2:GPU1` | 10 | runtime OOM, 9/10 running |

The module42 OOM occurred in the RE-SAC replay-buffer scatter path after real
training progress, not during scheduler placement. This makes profile 10 a
live robust capacity boundary for the current `hybrid_rl_resac_ant` bucket.

Implementation changes:

```text
simulation/service_cache.py
simulation/defaults.py
simulation/tasksets.py
```

The cache now treats a capacity-boundary profile as an upper cap for that
workload: profile `k` and all profiles above `k` are excluded from feasible
replay profiles. Repeated valid measurements now keep the conservative lower
aggregate rate rather than the historical maximum.

## Q11 Fresh Robust Slices

Fresh live slices after the profile-10 boundary:

| Run | Profile | Placement | Boundary | Aggregate iter/s |
|---|---:|---|---|---:|
| `module43_q11_live_sanity_jtl110gpu2_gpu1_profile9_20260608_001` | 9 | valid | no | 0.312017 |
| `module44_q11_live_sanity_jtl110gpu2_gpu1_profile1_20260608_001` | 1 | valid | no | 0.253059 |
| `module45_q11_live_sanity_jtl110gpu2_gpu1_profile8_20260608_001` | 8 | valid | no | 0.312533 |
| `module46_q11_live_sanity_jtl110gpu2_gpu1_profiles2_3_20260608_001` | 2 | valid | no | 0.338983 |
| `module46_q11_live_sanity_jtl110gpu2_gpu1_profiles2_3_20260608_001` | 3 | valid | no | 0.337079 |

With the robust boundary and lower-service cache, standalone q11 selects
profile 2/GPU. The mixed portfolio selects profile 3/GPU because the global
selector is support-first: it minimizes global makespan/support before spending
slack on mean flow.

## Q11 Candidate Live Validation

Replay report:

```text
/tmp/q11_candidate_profile2_replay_report.json
```

Live report and validation:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module46_q11_live_sanity_jtl110gpu2_gpu1_profiles2_3_20260608_001/reports/q11_candidate_profile2_live_report.json
/home/erzhu419/.claude/scheduler/experiments/runs/module46_q11_live_sanity_jtl110gpu2_gpu1_profiles2_3_20260608_001/reports/q11_candidate_profile2_live_validation.json
```

| Metric | Replay predicted | Fresh live proxy | Relative error |
|---|---:|---:|---:|
| makespan_s | 38382.365 | 37760.000 | 0.016215 |
| mean_flow_s | 19394.382 | 19116.000 | 0.014354 |
| p90_flow_s | 34574.884 | 33984.000 | 0.017090 |

`usable_for_live_sanity = true` at 20% relative-error tolerance.

## Replay/SOTA After Robust Boundary

Standalone q11:

```text
candidate profile: hybrid_rl_resac_ant = 2
legacy profile:    hybrid_rl_resac_ant = 5
makespan improvement vs legacy: 1.1536x
mean-flow improvement vs legacy: 1.1758x
SOTA-style Pareto dominated: no
```

Hybrid portfolio:

```text
candidate profiles: cpu_heavy_local_bench=8, gpu_heavy_jax_matmul=1, hybrid_rl_resac_ant=3
makespan improvement vs legacy: 1.1609x
mean-flow improvement vs legacy: 1.2688x
SOTA-style Pareto dominated: no
```

Fixed SOTA-policy matrix:

```text
aggregate_candidate_not_pareto_dominated = true
```

Interpretation: the robust q11 boundary reduces the old profile-10 headline
speed, but it removes an infeasible action and keeps Scheduleurm on the Pareto
front against the SOTA-style policies under the same measured service cache.
