# Module29 Explicit Task-List Benchmark

Date: 2026-06-05

This module converts the replay benchmark into the reviewer-facing form:

```text
one task list per quadrant;
the same task list is replayed by legacy, Scheduleurm candidate, and SOTA-style policies;
each policy reports all-job completion time, mean flow time, and p90 flow time.
```

This matches the common systems-scheduler pattern documented in module21:
measure a service/goodput table on the real fabric, build static or arrival-time
traces, then replay policy semantics on the same trace. Gavel/Pollux/Sia-like
baselines use the measured table; SRPT/Gittins-like baselines use finite-batch
delay oracles; IADeep/Salus-like baselines use guarded co-location semantics.

## Artifact

The new layer is separate from `scheduler.py`:

```text
simulation/trace_benchmark.py
simulation/trace_benchmark_cli.py
skill/tests/test_trace_benchmark.py
```

It does not replace the aggregate fast-forward replay. It adds a stricter
benchmark surface: every method receives the exact same explicit jobs.

## Task Lists

The task lists are generated from `simulation/tasksets.py`.

| Taskset | Quadrant | Jobs | Resource pool | Status |
|---|---|---:|---:|---|
| `q00_light_control` | low CPU, low GPU | 512 | 1 local control pool | real local curve |
| `q01_gpu_bound_compute` | low CPU, high GPU | 48 | 2 GPUs | real profiles 1-8 |
| `q10_cpu_host_bound` | high CPU, low GPU | 256 | 1 host pool | protocol curve |
| `q11_cpu_gpu_coupled` | high CPU, high GPU | 160 | 1 GPU | real profiles 1-12, closed at 13 |
| `hybrid_research_portfolio` | mixed | 400 | hybrid/GPU/CPU pools | current portfolio |

`static` arrival mode is the saturation benchmark: all jobs are present at time
zero, so `makespan_s` is exactly "how long until all n tasks finish." `poisson`
arrival mode is also implemented, but it is for later stability/load sweeps; it
should not replace the static batch table as the first performance claim.

## Commands

```bash
python3 -m simulation.trace_benchmark_cli --taskset q00_light_control --arrival-mode static --seed 42 --replay-seed 7
python3 -m simulation.trace_benchmark_cli --taskset q01_gpu_bound_compute --arrival-mode static --seed 42 --replay-seed 7
python3 -m simulation.trace_benchmark_cli --taskset q10_cpu_host_bound --arrival-mode static --seed 42 --replay-seed 7
python3 -m simulation.trace_benchmark_cli --taskset q11_cpu_gpu_coupled --arrival-mode static --seed 42 --replay-seed 7
python3 -m simulation.trace_benchmark_cli --taskset hybrid_research_portfolio --arrival-mode static --seed 42 --replay-seed 7
```

Fixed-policy matrix command:

```bash
python3 -m simulation.trace_benchmark_cli \
  --matrix \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7 \
  --report-out /tmp/scheduleurm_tasklist_matrix.json
```

Optional trace export:

```bash
python3 -m simulation.trace_benchmark_cli \
  --taskset q01_gpu_bound_compute \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7 \
  --trace-out /tmp/q01_trace.json \
  --report-out /tmp/q01_trace_report.json
```

The exported trace contains every job id, workload key, resource kind, arrival
time, total work units, and resource-pool count. The report contains each
policy's selected profile and completion metrics.

The pass condition is not legacy-only:

```text
pass =
  candidate improves both all-job makespan and mean flow over Scheduleurm legacy;
  candidate is not Pareto-dominated by any SOTA-style baseline
  on all-job makespan and mean flow.
```

## Static Task-List Results

Scheduleurm candidate vs legacy:

| Taskset | Candidate profile | All-job makespan improvement | Mean-flow improvement | p90-flow improvement |
|---|---|---:|---:|---:|
| `q00_light_control` | `light_control_local=8` | 7.887x | 7.887x | 7.905x |
| `q01_gpu_bound_compute` | `gpu_heavy_jax_matmul=4` | 1.079x | 1.045x | 1.085x |
| `q10_cpu_host_bound` | `cpu_heavy_protocol=16` | 1.281x | 1.387x | 1.380x |
| `q11_cpu_gpu_coupled` | `hybrid_rl_resac_ant=10` | 1.229x | 1.203x | 1.205x |
| `hybrid_research_portfolio` | `cpu=16, gpu=1, hybrid=1` | 1.284x | 1.360x | 1.358x |

Scheduleurm candidate vs SOTA-style baselines on the same explicit task lists:

| Taskset | Strongest SOTA-style comparison | Candidate vs SOTA makespan | Candidate vs SOTA mean flow | Pareto status |
|---|---|---:|---:|---|
| `q00_light_control` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q01_gpu_bound_compute` | throughput table endpoint | 0.984x | 1.116x | tradeoff, not dominated |
| `q01_gpu_bound_compute` | delay oracle endpoint | 1.014x | 0.911x | tradeoff, not dominated |
| `q10_cpu_host_bound` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q11_cpu_gpu_coupled` | delay oracle endpoint | 1.029x | 0.983x | tradeoff, not dominated |
| `hybrid_research_portfolio` | throughput table endpoint | 1.000x | 1.006x | candidate better mean-flow |
| `hybrid_research_portfolio` | interference/composite endpoint | 1.000x | 1.006x | candidate better mean-flow |
| `hybrid_research_portfolio` | delay oracle endpoint | 1.000x | 1.000x | tie |

## Fixed SOTA-Policy Matrix

The previous table is per-taskset. This subsection answers the stricter
question: if a SOTA-style policy is best on q00, q01, q10, or q11, can that
same fixed policy complete every other task list better than Scheduleurm?

The answer is no under the current measured service cache. No individual
SOTA-style policy Pareto-dominates Scheduleurm candidate on total makespan and
job-weighted mean flow.

Interpretation of ratio columns:

```text
candidate vs policy > 1: Scheduleurm candidate is faster/better.
candidate vs policy < 1: that fixed SOTA-style policy is faster/better.
```

Four-quadrant suite, excluding portfolio to avoid double-counting the mixed
portfolio jobs:

| Fixed policy | Representative systems | Jobs | Sum makespan (s) | Job-weighted mean flow (s) | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---:|---:|---:|---:|---:|
| Scheduleurm candidate | this work | 976 | 110844.614 | 14480.992 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 976 | 110817.316 | 14486.506 | 0.9998x | 1.0004x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 976 | 111917.637 | 14425.051 | 1.0097x | 0.9961x |
| IADeep/Salus-style interference guard | IADeep, Salus | 976 | 110844.614 | 14480.992 | 1.0000x | 1.0000x |
| quadrant composite | mixed SOTA-style | 976 | 110844.614 | 14480.992 | 1.0000x | 1.0000x |

Full five-taskset suite, including `hybrid_research_portfolio`:

| Fixed policy | Representative systems | Jobs | Sum makespan (s) | Job-weighted mean flow (s) | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---:|---:|---:|---:|---:|
| Scheduleurm candidate | this work | 1376 | 170851.072 | 17238.807 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 1376 | 170823.774 | 17285.309 | 0.9998x | 1.0027x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 1376 | 171924.095 | 17199.128 | 1.0063x | 0.9977x |
| IADeep/Salus-style interference guard | IADeep, Salus | 1376 | 170851.072 | 17281.675 | 1.0000x | 1.0025x |
| quadrant composite | mixed SOTA-style | 1376 | 170851.072 | 17281.675 | 1.0000x | 1.0025x |

Winner-transfer view:

| Source taskset | Winner objective | Best fixed SOTA-style policy on source | Candidate vs that policy on full five-taskset makespan | Candidate vs that policy on full five-taskset flow |
|---|---|---|---:|---:|
| q00 | makespan or mean-flow | all SOTA-style policies tie; first-listed throughput table shown | 0.9998x | 1.0027x |
| q01 | makespan | throughput table | 0.9998x | 1.0027x |
| q01 | mean-flow | delay oracle | 1.0063x | 0.9977x |
| q10 | makespan or mean-flow | all SOTA-style policies tie; first-listed throughput table shown | 0.9998x | 1.0027x |
| q11 | makespan | throughput/interference/composite tie; first-listed throughput table shown | 0.9998x | 1.0027x |
| q11 | mean-flow | delay oracle | 1.0063x | 0.9977x |

q01 SOTA endpoint check on the same 48-job task list:

| Policy | Profile | Makespan (s) | Mean flow (s) | Interpretation |
|---|---|---:|---:|---|
| legacy | 3/GPU | 1792.164 | 1007.326 | old fixed cap |
| Scheduleurm candidate | 4/GPU | 1661.313 | 963.498 | guarded Pareto knee |
| Gavel/Pollux/Sia-style throughput table | 8/GPU | 1634.016 | 1075.617 | best throughput endpoint |
| SRPT/Gittins-style delay oracle | 1/GPU | 1684.904 | 877.959 | best delay endpoint |
| IADeep/Salus-style interference guard | 4/GPU | 1661.313 | 963.498 | matches candidate on q01 |

This confirms the intended Pareto geometry rather than hiding it:

```text
candidate beats legacy on all-job time and flow;
candidate gives up about 1.7% all-job time to the throughput endpoint;
candidate gains about 11.6% mean-flow over the throughput endpoint;
candidate beats the delay endpoint on all-job time but loses mean-flow.
```

Portfolio check on the same 400-job list:

| Policy | Profiles | Makespan (s) | Mean flow (s) |
|---|---|---:|---:|
| legacy | `cpu=32, gpu=3, hybrid=5` | 77064.735 | 32592.348 |
| Scheduleurm candidate | `cpu=16, gpu=1, hybrid=1` | 60006.458 | 23967.877 |
| Gavel/Pollux/Sia-style throughput table | `cpu=16, gpu=6, hybrid=10` | 60006.458 | 24114.389 |
| SRPT/Gittins-style delay oracle | `cpu=16, gpu=1, hybrid=1` | 60006.458 | 23967.877 |
| IADeep/Salus-style interference guard | `cpu=16, gpu=1, hybrid=10` | 60006.458 | 24115.341 |

The portfolio candidate keeps the same all-job time as the best SOTA-style
endpoints and improves mean flow against throughput/interference policies.

## Validation Scope

This module validates the benchmark form, not a new theorem. It makes the
experiment line cleaner:

```text
proof route: support/candidate-set/robust stability;
math route: policy classes and constants;
experiment route: measured service table + explicit task-list replay.
```

The remaining experimental gap is not benchmark shape; it is empirical breadth:
replace the q10 protocol curve with a real CPU/data-loader-heavy trace, add more
arrival seeds, and run longer stability/load sweeps after the module-level
policy validations stay passing.
