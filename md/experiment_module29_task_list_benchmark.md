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
| `q00_light_control` | low CPU, low GPU | 512 | 1 local control pool | real profiles 1-13, closed at 14 |
| `q01_gpu_bound_compute` | low CPU, high GPU | 48 | 2 GPUs | real profiles 1-8 |
| `q10_cpu_host_bound` | high CPU, low GPU | 256 | 1 local CPU bucket | real profiles 1-9, closed at 10 |
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
| `q00_light_control` | `light_control_local=13` | 11.825x | 11.758x | 11.814x |
| `q01_gpu_bound_compute` | `gpu_heavy_jax_matmul=4` | 1.079x | 1.045x | 1.085x |
| `q10_cpu_host_bound` | `cpu_heavy_local_bench=8` | 1.510x | 1.534x | 1.534x |
| `q11_cpu_gpu_coupled` | `hybrid_rl_resac_ant=10` | 1.229x | 1.203x | 1.205x |
| `hybrid_research_portfolio` | `cpu=8, gpu=1, hybrid=10` | 1.228x | 1.283x | 1.201x |

Scheduleurm candidate vs SOTA-style baselines on the same explicit task lists:

| Taskset | Strongest SOTA-style comparison | Candidate vs SOTA makespan | Candidate vs SOTA mean flow | Pareto status |
|---|---|---:|---:|---|
| `q00_light_control` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q01_gpu_bound_compute` | throughput table endpoint | 0.984x | 1.116x | tradeoff, not dominated |
| `q01_gpu_bound_compute` | delay oracle endpoint | 1.014x | 0.911x | tradeoff, not dominated |
| `q10_cpu_host_bound` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q11_cpu_gpu_coupled` | delay oracle endpoint | 1.029x | 0.983x | tradeoff, not dominated |
| `hybrid_research_portfolio` | throughput table endpoint | 1.001x | 0.999x | tradeoff, not dominated |
| `hybrid_research_portfolio` | interference/composite endpoint | 1.000x | 1.000x | tie |
| `hybrid_research_portfolio` | delay oracle endpoint | 1.028x | 0.977x | tradeoff, not dominated |

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
| Scheduleurm candidate | this work | 976 | 50937.535 | 6016.439 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 976 | 50910.238 | 6021.953 | 0.9995x | 1.0009x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 976 | 52010.559 | 5960.498 | 1.0211x | 0.9907x |
| IADeep/Salus-style interference guard | IADeep, Salus | 976 | 50937.535 | 6016.439 | 1.0000x | 1.0000x |
| quadrant composite | mixed SOTA-style | 976 | 50937.535 | 6016.439 | 1.0000x | 1.0000x |

Full five-taskset suite, including `hybrid_research_portfolio`:

| Fixed policy | Representative systems | Jobs | Sum makespan (s) | Job-weighted mean flow (s) | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---:|---:|---:|---:|---:|
| Scheduleurm candidate | this work | 1376 | 77849.461 | 5989.780 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 1376 | 77858.340 | 5992.094 | 1.0001x | 1.0004x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 1376 | 79673.134 | 5911.037 | 1.0234x | 0.9869x |
| IADeep/Salus-style interference guard | IADeep, Salus | 1376 | 77849.461 | 5989.780 | 1.0000x | 1.0000x |
| quadrant composite | mixed SOTA-style | 1376 | 77849.461 | 5989.780 | 1.0000x | 1.0000x |

Winner-transfer view:

| Source taskset | Winner objective | Best fixed SOTA-style policy on source | Candidate vs that policy on full five-taskset makespan | Candidate vs that policy on full five-taskset flow |
|---|---|---|---:|---:|
| q00 | makespan or mean-flow | all SOTA-style policies tie; first-listed throughput table shown | 1.0001x | 1.0004x |
| q01 | makespan | throughput table | 1.0001x | 1.0004x |
| q01 | mean-flow | delay oracle | 1.0234x | 0.9869x |
| q10 | makespan or mean-flow | throughput table | 1.0001x | 1.0004x |
| q11 | makespan | throughput/interference/composite tie; first-listed throughput table shown | 1.0001x | 1.0004x |
| q11 | mean-flow | delay oracle | 1.0234x | 0.9869x |

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
| legacy | `cpu=9, gpu=3, hybrid=5` | 33043.785 | 7599.635 |
| Scheduleurm candidate | `cpu=8, gpu=1, hybrid=10` | 26911.926 | 5924.731 |
| Gavel/Pollux/Sia-style throughput table | `cpu=8, gpu=6, hybrid=10` | 26948.103 | 5919.239 |
| SRPT/Gittins-style delay oracle | `cpu=8, gpu=1, hybrid=1` | 27662.575 | 5790.350 |
| IADeep/Salus-style interference guard | `cpu=8, gpu=1, hybrid=10` | 26911.926 | 5924.731 |

The portfolio candidate is now based on a real q10 local CPU curve rather than
the old protocol CPU curve. It ties the interference/composite policy, has a
small makespan edge but small mean-flow loss against the throughput endpoint,
and has the opposite tradeoff against the delay endpoint.

## Validation Scope

This module validates the benchmark form, not a new theorem. It makes the
experiment line cleaner:

```text
proof route: support/candidate-set/robust stability;
math route: policy classes and constants;
experiment route: measured service table + explicit task-list replay.
```

The remaining experimental gap is not benchmark shape. q10 is now real for the
declared local CPU bucket; remaining breadth is a remote CPU-node/data-loader
replication, more arrival seeds, and longer stability/load sweeps after the
module-level policy validations stay passing.
