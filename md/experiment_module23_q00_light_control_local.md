# Q00 Light-Control Local Validation

Date: 2026-06-04

Current status: this module is the historical seed curve for profiles 1-8.
Module38 extends the same exact workload through profiles 9-13 and records
profile 14 as the active local scheduling-capacity boundary. Do not treat the
profile 16 boundary below as the current q00 closure.

This module fills the `q00_light_control` taskset with a real CPU-only local
control curve. It measures a tiny progress-bearing command with very light work
and a short sleep per step. The goal is to validate scheduler overhead,
queueing, and low-resource concurrency without consuming GPUs.

The runner cancels tasks after a stable progress window. It does not wait for
natural completion.

## Commands

Clean profiles 1, 2, 4, and 8 were measured in:

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module23_q00_light_control_local_profiles1_16_20260604_001 \
  --node local \
  --profiles 1,2,4,8,16 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_cpu_progress_benchmark.py \
  --python-bin python3 \
  --steps 10000 \
  --mode light \
  --work-items 100 \
  --sleep-s 0.020 \
  --ram-mb 256 \
  --cpu 1 \
  --warmup-timeout-s 120 \
  --measure-s 45 \
  --poll-s 15 \
  --hard-rule-mode clean_bench \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/light_control
```

Profile 16 was then rerun with boundary recording:

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module23_q00_light_control_local_profile16_boundary_20260604_001 \
  --node local \
  --profiles 16 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_cpu_progress_benchmark.py \
  --python-bin python3 \
  --steps 10000 \
  --mode light \
  --work-items 100 \
  --sleep-s 0.020 \
  --ram-mb 256 \
  --cpu 1 \
  --warmup-timeout-s 120 \
  --measure-s 45 \
  --poll-s 15 \
  --hard-rule-mode clean_bench \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/light_control \
  --stop-on-capacity-boundary
```

Exact replay of the 8-way profile needs every intermediate active count in the
tail wave. Profiles 3, 5, 6, and 7 were therefore measured in:

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module24_q00_light_control_local_profiles3_7_20260604_001 \
  --node local \
  --profiles 3,5,6,7 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_cpu_progress_benchmark.py \
  --python-bin python3 \
  --steps 10000 \
  --mode light \
  --work-items 100 \
  --sleep-s 0.020 \
  --ram-mb 256 \
  --cpu 1 \
  --warmup-timeout-s 120 \
  --measure-s 45 \
  --poll-s 15 \
  --hard-rule-mode clean_bench \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/light_control
```

## Result

Run directories:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module23_q00_light_control_local_profiles1_16_20260604_001
/home/erzhu419/.claude/scheduler/experiments/runs/module23_q00_light_control_local_profile16_boundary_20260604_001
/home/erzhu419/.claude/scheduler/experiments/runs/module24_q00_light_control_local_profiles3_7_20260604_001
```

| Profile | Running | With rate | Median aggregate step/s | Boundary | Blocked/queued |
|---:|---:|---:|---:|:---|---:|
| 1 | 1 | 1 | 51.446729 | no | 0 |
| 2 | 2 | 2 | 100.577699 | no | 0 |
| 3 | 3 | 3 | 148.667625 | no | 0 |
| 4 | 4 | 4 | 205.816828 | no | 0 |
| 5 | 5 | 5 | 251.982663 | no | 0 |
| 6 | 6 | 6 | 303.518782 | no | 0 |
| 7 | 7 | 7 | 354.609844 | no | 0 |
| 8 | 8 | 8 | 410.595780 | no | 0 |
| 16 | 11 | 11 | 564.897210 | yes | 5 |

The profile 16 result is a local scheduling-capacity boundary, not a usable
service-curve point. In the boundary rerun, 11 tasks reached running/progress
and 5 remained queued until warmup timeout.

## Interpretation

For local light-control work, profiles 1-8 scale almost linearly.
Profile 16 was not feasible in this historical run. Module38 subsequently
filled the missing tail with the same command parameters and closed q00 more
tightly at profile 14.

This should be reported as a local control-plane bucket result. It should not be
generalized to remote CPU nodes until a remote CPU node hostname/probe is
available and measured separately.
