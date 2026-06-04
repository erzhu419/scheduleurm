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

## Static Task-List Results

Scheduleurm candidate vs legacy:

| Taskset | Candidate profile | All-job makespan improvement | Mean-flow improvement | p90-flow improvement |
|---|---|---:|---:|---:|
| `q00_light_control` | `light_control_local=8` | 7.887x | 7.887x | 7.905x |
| `q01_gpu_bound_compute` | `gpu_heavy_jax_matmul=4` | 1.079x | 1.045x | 1.085x |
| `q10_cpu_host_bound` | `cpu_heavy_protocol=16` | 1.281x | 1.387x | 1.380x |
| `q11_cpu_gpu_coupled` | `hybrid_rl_resac_ant=10` | 1.229x | 1.203x | 1.205x |
| `hybrid_research_portfolio` | `cpu=16, gpu=1, hybrid=1` | 1.284x | 1.360x | 1.358x |

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
candidate gains about 11.0% mean-flow over the throughput endpoint;
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
