# Module26 Guarded Statewise Replay Optimization

Date: 2026-06-04

This module upgrades the replay candidate from a pure makespan selector to a
resource-class guarded statewise selector. The baseline in this module is
Scheduleurm legacy fixed caps, not an external SOTA scheduler. External systems
such as Gavel, Pollux, Sia, Tiresias, THEMIS, IADeep, Decima, and Salus are
currently reference material and future baselines, not active competitors in the
reported pass/fail numbers.

## Problem

The makespan-only calibrated policy selected:

```text
q01 GPU-heavy: 8/GPU
q11 hybrid RL: 10/GPU
portfolio GPU-heavy member: 6/GPU
```

That was throughput-rational for q01 but incomplete as a scheduler claim:
standalone q01 makespan improved by 1.096x, while q01 mean flow fell to 0.940x
of legacy. The cause is visible in the real service curve: q01 aggregate service
is almost flat across high profiles, while per-task service collapses as
co-location increases.

## Mathematical Alignment

The selector implements the proof route as:

```text
support/service objective first
bounded delay/congestion penalty second
statewise queue-scaled penalty only inside the calibrated guard
```

For a workload class and measured profile `k`, replay computes:

```text
M_k(n) = deterministic all-task makespan proxy
F_k(n) = deterministic mean-flow proxy
```

The guarded selector first finds `M* = min_k M_k(n)`, then admits only profiles
with:

```text
M_k(n) <= (1 + eps_guard(n)) M*
```

Among admitted profiles it chooses the smallest `F_k(n)`, then lower makespan,
then lower profile id. For pure GPU-heavy statewise replay:

```text
eps_guard(n) = 0.02 + 0.6 / n
```

For hybrid RL, CPU protocol, and light control the default candidate keeps the
service/makespan support objective. This is a regime-bucket decision, not a
workload-key shortcut: `resource_kind="gpu_heavy"` is the only class using the
queue-scaled guarded statewise drain in this module.

This maps to the Lean/math line:

```text
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

The makespan guard is the approximate support/oracle loss. The mean-flow
tie-break is the bounded penalty. Because the penalty is only allowed inside the
guard, it does not replace the MaxWeight/support term.

## Code

Changed files:

```text
simulation/service_cache.py
simulation/fast_forward.py
simulation/defaults.py
skill/tests/test_simulation_fast_forward.py
```

New or changed APIs:

```text
deterministic_mean_flow_s(...)
ServiceRateCache.best_profile_for_guarded_mean_flow(...)
ReplayPolicy.guarded_resource_kinds
ReplayPolicy.statewise_resource_kinds
calibrated_policy() -> calibrated_guarded_statewise
calibrated_makespan_policy() -> makespan-only ablation
```

Replay reports now include `policy_config`, `profile_trace_len`, and
`profile_trace_counts`, so the guard/statewise settings and realized profile
trajectory are visible in the JSON output.

Default candidate:

```text
hybrid_rl_resac_ant: makespan/support objective, selected 10/GPU
gpu_heavy_jax_matmul: guarded statewise objective, selected 1/GPU
cpu_heavy_protocol: makespan/support objective, selected 16 workers
light_control_local: makespan/support objective, selected 8 workers
```

## Validation

Commands:

```bash
python3 -m py_compile simulation/fast_forward.py simulation/service_cache.py simulation/defaults.py skill/tests/test_simulation_fast_forward.py

python3 -m simulation.cli --taskset q01_gpu_bound_compute --trials 31 --seed 42 --min-makespan-improvement 1.03 --min-class-improvement 1.03

python3 -m simulation.cli --taskset q11_cpu_gpu_coupled --trials 31 --seed 42 --min-makespan-improvement 1.03 --min-class-improvement 1.03

python3 -m simulation.cli --taskset hybrid_research_portfolio --trials 31 --seed 42 --min-makespan-improvement 1.05 --min-class-improvement 1.03
```

Targeted test loader:

```text
checks=18 failed=0
```

Standalone tasksets:

| Taskset | Candidate profile | Makespan improvement | Mean-flow improvement | Pass |
|---|---:|---:|---:|---|
| `q00_light_control` | 8 | 7.929x | 7.887x | yes |
| `q01_gpu_bound_compute` | 1 | 1.064x | 1.148x | yes |
| `q11_cpu_gpu_coupled` | 10 | 1.228x | 1.207x | yes |

Portfolio:

| Metric | Legacy | Candidate | Improvement |
|---|---:|---:|---:|
| Total makespan | 76472.987 s | 59534.263 s | 1.285x |
| Weighted mean flow | 32431.821 s | 23960.208 s | 1.354x |

Portfolio per workload:

| Workload | Legacy profile | Candidate profile | Makespan improvement | Mean-flow improvement |
|---|---:|---:|---:|---:|
| `hybrid_rl_resac_ant` | 5 | 10 | 1.227x | 1.190x |
| `gpu_heavy_jax_matmul` | 3 | 1 | 1.064x | 1.225x |
| `cpu_heavy_protocol` | 32 | 16 | 1.285x | 1.390x |

Against the makespan-only ablation:

```text
q01 makespan-only: makespan 1.096x, mean-flow 0.940x, profile 8
q01 guarded statewise: makespan 1.064x, mean-flow 1.148x, profile 1

portfolio makespan-only: makespan 1.285x, mean-flow 1.353x
portfolio guarded statewise: makespan 1.285x, mean-flow 1.354x
```

## Reviewer-Facing Status

This module validates an algorithmic improvement over Scheduleurm legacy in the
trace-driven replay layer. Module27 adds SOTA-style policy-semantics replay
baselines on the same measured service cache. Direct external-scheduler
superiority still needs one of:

```text
external scheduler implementation run on the same traces and measured service cache
or a faithful replay baseline reproducing a paper's policy semantics
```

Do not call this a direct external-SOTA result until that layer exists.
