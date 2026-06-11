# Module29 Explicit Task-List Benchmark

Date: 2026-06-05

Update on 2026-06-11: modules39-46 replace the old q11 profile-10 candidate
with a live-robust q11 feasible family. Standalone q11 now selects profile 2
and the mixed portfolio selects profile 3. The tables below are the current
robust static replay; old profile-10 rows are historical module29 provenance
only and must not be used as current claims.

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

This wording is deliberate.  The benchmark compares SOTA-style policy semantics
on the same measured Scheduleurm service cache.  It does not claim direct
binary execution of Gavel, Pollux, Sia, IADeep, or Salus.

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
| `q11_cpu_gpu_coupled` | high CPU, high GPU | 160 | 1 GPU | real profiles 1-9, closed at 10 |
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
| `q11_cpu_gpu_coupled` | `hybrid_rl_resac_ant=2` | 1.154x | 1.176x | 1.156x |
| `hybrid_research_portfolio` | `cpu=8, gpu=1, hybrid=3` | 1.161x | 1.269x | 1.186x |

Scheduleurm candidate vs SOTA-style baselines on the same explicit task lists:

| Taskset | Strongest SOTA-style comparison | Candidate vs SOTA makespan | Candidate vs SOTA mean flow | Pareto status |
|---|---|---:|---:|---|
| `q00_light_control` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q01_gpu_bound_compute` | throughput table endpoint | 0.984x | 1.116x | tradeoff, not dominated |
| `q01_gpu_bound_compute` | delay oracle endpoint | 1.014x | 0.911x | tradeoff, not dominated |
| `q10_cpu_host_bound` | all SOTA-style policies tie candidate | 1.000x | 1.000x | not dominated |
| `q11_cpu_gpu_coupled` | all SOTA-style policies tie candidate | 1.000x | 1.000x | tie |
| `hybrid_research_portfolio` | throughput table endpoint | 1.000x | 1.001x | tradeoff, not dominated |
| `hybrid_research_portfolio` | delay/interference/composite endpoint | 1.005x | 0.997x | tradeoff, not dominated |

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
| Scheduleurm candidate | this work | 976 | 53310.215 | 6089.605 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 976 | 53282.917 | 6095.119 | 0.9995x | 1.0009x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 976 | 53333.806 | 6085.398 | 1.0004x | 0.9993x |
| IADeep/Salus-style interference guard | IADeep, Salus | 976 | 53310.215 | 6089.605 | 1.0000x | 1.0000x |
| quadrant composite | mixed SOTA-style | 976 | 53310.215 | 6089.605 | 1.0000x | 1.0000x |

Full five-taskset suite, including `hybrid_research_portfolio`:

| Fixed policy | Representative systems | Jobs | Sum makespan (s) | Job-weighted mean flow (s) | Candidate vs policy makespan | Candidate vs policy flow |
|---|---|---:|---:|---:|---:|---:|
| Scheduleurm candidate | this work | 1376 | 81818.460 | 6060.620 | 1.0000x | 1.0000x |
| Gavel/Pollux/Sia-style throughput table | Gavel, Pollux, Sia | 1376 | 81793.703 | 6067.049 | 0.9997x | 1.0011x |
| SRPT/Gittins-style delay oracle | SRPT, Gittins, SERPT | 1376 | 81988.550 | 6052.181 | 1.0021x | 0.9986x |
| IADeep/Salus-style interference guard | IADeep, Salus | 1376 | 81964.959 | 6055.165 | 1.0018x | 0.9991x |
| quadrant composite | mixed SOTA-style | 1376 | 81964.959 | 6055.165 | 1.0018x | 0.9991x |

Winner-transfer view:

| Source taskset | Winner objective | Best fixed SOTA-style policy on source | Candidate vs that policy on full five-taskset makespan | Candidate vs that policy on full five-taskset flow |
|---|---|---|---:|---:|
| q00 | makespan or mean-flow | all SOTA-style policies tie; first-listed throughput table shown | 0.9997x | 1.0011x |
| q01 | makespan | throughput table | 0.9997x | 1.0011x |
| q01 | mean-flow | delay oracle | 1.0021x | 0.9986x |
| q10 | makespan or mean-flow | throughput table | 0.9997x | 1.0011x |
| q11 | makespan or mean-flow | all SOTA-style policies tie; first-listed throughput table shown | 0.9997x | 1.0011x |

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
| legacy | `cpu=9, gpu=3, hybrid=5` | 33095.049 | 7599.763 |
| Scheduleurm candidate | `cpu=8, gpu=1, hybrid=3` | 28508.245 | 5989.897 |
| Gavel/Pollux/Sia-style throughput table | `cpu=8, gpu=6, hybrid=3` | 28510.786 | 5998.559 |
| SRPT/Gittins-style delay oracle | `cpu=8, gpu=1, hybrid=2` | 28654.744 | 5971.132 |
| IADeep/Salus-style interference guard | `cpu=8, gpu=1, hybrid=2` | 28654.744 | 5971.132 |

The portfolio candidate is now based on the real q10 local CPU curve and the
live-robust q11 boundary. It is faster in all-job completion time than the
delay/interference/composite endpoints, while the throughput endpoint is
essentially tied on makespan and slightly worse on mean flow under this static
task-list replay.

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

The q00, q10, and q11 tasksets should be cited as declared bucket certificates:
`q00_light_control` for local light-control work and `q10_cpu_host_bound` for
the local CPU-heavy benchmark bucket, plus `q11_cpu_gpu_coupled` for the current
`hybrid_rl_resac_ant` robust node bucket. They are not evidence that every
CPU-heavy, data-loader-heavy, RL, or BAPR deployment has the same service curve.
