# Module31 q10 Real CPU Trace Closure

Date: 2026-06-05

This module records the current status of the q10 high-CPU/low-GPU quadrant.
The active `q10_cpu_host_bound` benchmark still uses `cpu_heavy_protocol`; that
must not be described as a theorem-grade real CPU trace. However, a real local
CPU-heavy progress-bearing curve has now been connected to the service cache as
a separate probe taskset.

## New Real Probe Taskset

```text
q10_cpu_host_bound_local_real_probe
```

Workload key:

```text
cpu_heavy_local_bench
```

Source run:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module25_q10_cpu_heavy_local_profiles1_8_20260604_001
```

The default service cache now reads the module25 summaries for profiles 1-8.
The taskset registry exposes the probe without replacing the active q10
protocol benchmark.

## Measured Curve

| Profile | Median aggregate step/s | Mean step/s/task |
|---:|---:|---:|
| 1 | 6.881652 | 6.881652 |
| 2 | 13.190345 | 6.595173 |
| 3 | 21.299104 | 7.099701 |
| 4 | 26.196447 | 6.549112 |
| 5 | 30.930518 | 6.186104 |
| 6 | 42.836850 | 7.139475 |
| 7 | 50.301283 | 7.185898 |
| 8 | 54.779604 | 6.847450 |

All profiles are progress-bearing and replayable. They are local-bucket
measurements, not yet a final cluster q10 comparison.

## Why This Does Not Fully Close q10 Yet

The missing piece is a same-bucket legacy-comparable cap. The active protocol
q10 benchmark compares against a 32-worker legacy cap. The local real run only
measures profiles 1-8. It would be misleading to claim that the local 8-way
profile beats or replaces an unmeasured 32-way local legacy cap.

Promotion requirements:

```text
1. Either declare local CPU as the official q10 bucket, or run the same command
   on a stable CPU-node bucket.
2. Measure the legacy-comparable cap on that same bucket.
3. Repeat or lengthen the profile windows enough to reduce local frequency and
   scheduler-noise concerns.
4. Promote q10_cpu_host_bound only after those same-bucket measurements exist.
```

## Commands

List tasksets:

```bash
python3 -m simulation.trace_benchmark_cli --list-tasksets
```

Run the real local q10 probe replay:

```bash
python3 -m simulation.trace_benchmark_cli \
  --taskset q10_cpu_host_bound_local_real_probe \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7
```

The output is a calibration/probe artifact. It should not replace the active
q10 theorem-grade result until the promotion requirements above are satisfied.
