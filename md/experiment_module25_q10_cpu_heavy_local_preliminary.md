# Q10 CPU-Heavy Local Preliminary Curve

Date: 2026-06-04

This module measures a real CPU-heavy, GPU-free command on the local node. It is
the preliminary half of the q10 closure. At the time of this module, q10 was
not promoted because only local profiles 1-8 had been measured. Module33 later
measured profile 9 and closed profile 10 as a capacity boundary; Module31 then
promoted the real local CPU bucket into the active `q10_cpu_host_bound` taskset.

## Command

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module25_q10_cpu_heavy_local_profiles1_8_20260604_001 \
  --node local \
  --profiles 1,2,3,4,5,6,7,8 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_cpu_progress_benchmark.py \
  --python-bin python3 \
  --steps 1000 \
  --mode cpu \
  --work-items 1000000 \
  --sleep-s 0 \
  --ram-mb 512 \
  --cpu 1 \
  --warmup-timeout-s 180 \
  --measure-s 45 \
  --poll-s 15 \
  --hard-rule-mode clean_bench \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/cpu_heavy_local
```

Run directory:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module25_q10_cpu_heavy_local_profiles1_8_20260604_001
```

## Result

Validation status: pass.

| Profile | Running | With rate | Median aggregate step/s | Mean step/s/task | Blocked |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 6.881652 | 6.881652 | 0 |
| 2 | 2 | 2 | 13.190345 | 6.595173 | 0 |
| 3 | 3 | 3 | 21.299104 | 7.099701 | 0 |
| 4 | 4 | 4 | 26.196447 | 6.549112 | 0 |
| 5 | 5 | 5 | 30.930518 | 6.186104 | 0 |
| 6 | 6 | 6 | 42.836850 | 7.139475 | 0 |
| 7 | 7 | 7 | 50.301283 | 7.185898 | 0 |
| 8 | 8 | 8 | 54.779604 | 6.847450 | 0 |

The curve is real and progress-bearing, but it is local-only. The profile 5 to
8 region is somewhat noisy, likely due to local CPU scheduling/frequency and
background state.

## Interpretation

This replaces speculation about whether a CPU-heavy command can be measured, but
by itself it did not close q10 as a scheduler-comparison taskset. The closure
now lives in `md/experiment_module31_q10_real_cpu_trace.md`.

Historical promotion requirements from this preliminary step:

- measure the same command on a stable CPU-node bucket, or make local the
  declared q10 bucket;
- measure the legacy-comparable cap on that same bucket, not a protocol-only
  32-worker curve;
- repeat at least one run or extend the measurement window to reduce local
  frequency/scheduling noise.

Current status after Module33/Module31:

```text
declared q10 bucket: local CPU-heavy progress-bearing command;
profiles 1-9: measured;
profile 10: measured capacity boundary;
legacy cap: profile 9;
candidate: profile 8.
```
