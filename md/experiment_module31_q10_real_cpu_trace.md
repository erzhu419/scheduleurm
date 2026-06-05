# Module31 q10 Real CPU Trace Closure

Date: 2026-06-05

This module records the current status of the q10 high-CPU/low-GPU quadrant.
The active `q10_cpu_host_bound` benchmark now uses the real local
`cpu_heavy_local_bench` curve. This closes q10 for the declared local CPU
bucket: profiles 1-9 are measured progress-bearing profiles, and profile 10 is
a measured clean capacity boundary. It still does not claim to close every
remote CPU-node or data-loader-heavy deployment bucket.

## Active q10 Taskset

```text
q10_cpu_host_bound
q10_cpu_host_bound_local_real_probe
```

Both tasksets point at the same workload key:

```text
cpu_heavy_local_bench
```

Source runs:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module25_q10_cpu_heavy_local_profiles1_8_20260604_001
/home/erzhu419/.claude/scheduler/experiments/runs/module33_q10_cpu_heavy_local_profiles9_32_20260605_001
```

The default service cache reads module25 for profiles 1-8 and module33 for
profile 9 plus the profile 10 capacity-boundary certificate. The legacy policy
for this local bucket is now profile 9, which is the highest measured feasible
packing level. The calibrated candidate selects profile 8.

## Measured Curve

| Profile | Status | Median aggregate step/s | Mean step/s/task | Source |
|---:|---|---:|---:|---|
| 1 | valid | 6.881652 | 6.881652 | module25 |
| 2 | valid | 13.190345 | 6.595173 | module25 |
| 3 | valid | 21.299104 | 7.099701 | module25 |
| 4 | valid | 26.196447 | 6.549112 | module25 |
| 5 | valid | 30.930518 | 6.186104 | module25 |
| 6 | valid | 42.836850 | 7.139475 | module25 |
| 7 | valid | 50.301283 | 7.185898 | module25 |
| 8 | valid | 54.779604 | 6.847450 | module25 |
| 9 | valid | 35.830257 | 3.981140 | module33 |
| 10 | capacity boundary | 0.000000 | 0.000000 | module33 |

The module33 boundary reason is important: profile 10 did not produce ten
running progress-bearing tasks on the local bucket. Nine tasks completed or ran
through warmup while one remained queued, so this is a scheduler-capacity
boundary for the declared local CPU bucket, not an inferred throughput drop.

## Promotion Decision

q10 is promoted because all same-bucket promotion requirements are now met:

```text
1. declared bucket: local CPU-heavy progress-bearing command;
2. measured feasible profiles: 1-9;
3. measured same-bucket legacy cap: profile 9;
4. measured candidate profile: profile 8;
5. measured closure boundary: profile 10;
6. active taskset uses the real curve rather than cpu_heavy_protocol.
```

The remaining breadth item is remote replication: repeat the same workload on a
remote CPU-node or data-loader-heavy bucket if the paper wants to claim q10
generality beyond local CPU pressure.

## Replay Result

On the static 256-job q10 task list:

| Policy | Profile | Makespan (s) | Mean flow (s) |
|---|---:|---:|---:|
| legacy local cap | 9 | 7184.671 | 3682.050 |
| Scheduleurm candidate | 8 | 4757.118 | 2400.597 |

Candidate improvement over the same-bucket legacy cap:

```text
all-job makespan: 1.510x
mean flow:        1.534x
p90 flow:         1.534x
```

All SOTA-style replay baselines currently choose the same profile 8 on this
single-bucket q10 task list, so q10 itself is a tie against those policy
semantics and a strict improvement against the legacy local cap.

## Commands

Run the active q10 replay:

```bash
python3 -m simulation.trace_benchmark_cli \
  --taskset q10_cpu_host_bound \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7
```

Run the explicit local probe alias:

```bash
python3 -m simulation.trace_benchmark_cli \
  --taskset q10_cpu_host_bound_local_real_probe \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7
```

Rerun the capacity-boundary probe when the local bucket changes:

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module33_q10_cpu_heavy_local_profiles9_32_20260605_001 \
  --node local \
  --profiles 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 \
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
  --signature-prefix ScheduleurmBench/cpu_heavy_local \
  --stop-on-capacity-boundary
```
