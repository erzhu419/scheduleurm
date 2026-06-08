# Module47 Portfolio Replay-To-Live Sanity

Date: 2026-06-08

This module closes the portfolio live-sanity obligation without waiting for
long production jobs to finish naturally.  It uses fresh progress-window live
slices for the GPU workloads, combines them with the declared real q10 local CPU
bucket, and replays the exact `hybrid_research_portfolio` task list with the
same completion-time semantics as `simulation.trace_benchmark`.

## Artifacts

```text
algorithm/experiments/portfolio_live_proxy.py
skill/tests/test_experiment_calibration.py
/home/erzhu419/.claude/scheduler/experiments/runs/module47_portfolio_live_sanity_q01_profile1_jtl110gpu_20260608_001/reports/portfolio_composed_live_report.json
/home/erzhu419/.claude/scheduler/experiments/runs/module47_portfolio_live_sanity_q01_profile1_jtl110gpu_20260608_001/reports/portfolio_live_validation.json
```

## Replay Candidate

Taskset:

```text
hybrid_research_portfolio, static arrivals, trace seed 42, replay seed 7
```

Candidate policy:

```text
calibrated_global_guarded
profiles:
  cpu_heavy_local_bench = 8
  gpu_heavy_jax_matmul  = 1
  hybrid_rl_resac_ant   = 3
```

Replay status before live composition:

```text
pass = true
makespan improvement vs legacy = 1.1609x
mean-flow improvement vs legacy = 1.2688x
p90-flow improvement vs legacy = 1.1860x
SOTA-style Pareto dominated = no
```

## Live Sources

| Workload | Profiles supplied | Source |
|---|---:|---|
| `gpu_heavy_jax_matmul` | 1 | fresh module47 q01 profile1 on `jtl110gpu` |
| `hybrid_rl_resac_ant` | 1 | fresh module44 q11 profile1 on `jtl110gpu2:GPU1` |
| `hybrid_rl_resac_ant` | 2, 3 | fresh module46 q11 profiles2-3 on `jtl110gpu2:GPU1` |
| `cpu_heavy_local_bench` | 1-8 | real local q10 curve from module25 |

The q01 profile1 live slice measured:

```text
aggregate rate across 2 GPUs = 69.299419 step/s
per-resource rate used by replay = 34.6497095 step/s/GPU
placement_valid = true
```

The q11 profile3 live slice measured:

```text
aggregate rate = 0.337078653 iter/s/GPU
placement_valid = true
```

The CPU-heavy member uses the declared local q10 benchmark bucket.  It is not
claiming remote CPU-node or data-loader generalization.

## Validation Result

Comparison file:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module47_portfolio_live_sanity_q01_profile1_jtl110gpu_20260608_001/reports/portfolio_live_validation.json
```

| Metric | Replay predicted | Composed live | Relative error |
|---|---:|---:|---:|
| makespan_s | 28508.245 | 28298.731 | 0.007349 |
| mean_flow_s | 5989.897 | 5957.450 | 0.005417 |
| p90_flow_s | 19206.782 | 19053.817 | 0.007964 |

```text
completed_jobs = 400
censored_jobs = 0
usable_for_live_sanity = true
max_relative_error threshold = 0.20
```

Per-workload live proxy summaries:

| Workload | Jobs | Mean flow (s) | Max flow (s) |
|---|---:|---:|---:|
| `cpu_heavy_local_bench` | 256 | 2426.996 | 4755.038 |
| `gpu_heavy_jax_matmul` | 24 | 451.283 | 836.922 |
| `hybrid_rl_resac_ant` | 120 | 14590.317 | 28298.731 |

## Interpretation

This is a replay-to-live sanity certificate, not a claim that the full 400-job
portfolio was run to natural completion.  The claim is narrower and stronger:
the same measured service slices, task list, candidate profiles, and completion
semantics predict the small live progress-window proxy within 1% on makespan,
mean flow, and p90 flow.  That is enough to close the artifact gap that replay
was detached from real Scheduleurm progress measurements.
