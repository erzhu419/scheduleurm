# Module38 Q00 Light-Control Extended Curve

Date: 2026-06-06

This module attacks the q00 low-CPU/low-GPU control quadrant after the first
local curve only covered profiles 1-8 and a sparse profile 16 boundary. It uses
the exact same workload parameters as modules23+24 and fills profiles 9-13,
with profile 14 recorded as the measured local scheduling-capacity boundary.

Earlier exploratory runs are not used as evidence:

| Run | Reason discarded |
|---|---|
| `module34_q00_light_control_local_profiles9_15_20260605_001` | short `steps=120`, no sleep; tasks completed before a stable warmup window |
| `module35_q00_light_control_local_profiles9_15_long_20260605_001` | no sleep and wrong workload shape |
| `module36_q00_light_control_local_profiles9_15_longrun_20260605_001` | no sleep and huge step rate; not the module23 q00 workload |
| `module37_q00_light_control_local_profiles9_15_sleep20ms_20260605_001` | correct sleep but wrong `work_items`, and local bucket was busy |

## Command

```bash
python3 -m algorithm.experiments.cpu_service_curve_validation \
  --run-id module38_q00_light_control_local_profiles9_15_exact_20260605_001 \
  --node local \
  --profiles 9,10,11,12,13,14,15 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_cpu_progress_benchmark.py \
  --python-bin python3 \
  --steps 10000 \
  --mode light \
  --work-items 100 \
  --sleep-s 0.02 \
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

Run directory:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module38_q00_light_control_local_profiles9_15_exact_20260605_001
```

## Service Curve Result

| Profile | Running | With rate | Median aggregate step/s | Boundary | Blocked/queued |
|---:|---:|---:|---:|:---|---:|
| 9 | 9 | 9 | 410.362502 | no | 0 |
| 10 | 10 | 10 | 488.405723 | no | 0 |
| 11 | 11 | 11 | 453.407884 | no | 0 |
| 12 | 12 | 12 | 556.580129 | no | 0 |
| 13 | 13 | 13 | 618.648332 | no | 0 |
| 14 | 13 progressed, 1 queued during warmup | 6 in final summary | 287.302883 active partial | yes | 1 |

Profile 14 is a capacity boundary because warmup timed out with 13 tasks having
made progress and the 14th still queued:

```text
profile_14_per_resource warmup timed out: progressed=13, required=14
```

The final summary's `running_count=6` is a post-timeout partial snapshot after
some short tasks had already finished. The boundary evidence is the warmup
status: 13 progressed, 1 queued. The service cache marks profile 14 unusable
for exact replay and uses profiles 1-13 only.

## Replay Impact

Static q00 task list, 512 jobs, same measured service cache:

| Policy | Profile | Makespan (s) | Mean flow (s) | p90 flow (s) |
|---|---:|---:|---:|---:|
| legacy fixed cap | 1 | 100620.207 | 50054.188 | 90542.858 |
| Scheduleurm candidate | 13 | 8509.418 | 4256.938 | 7664.060 |

Improvement over legacy:

| Metric | Ratio |
|---|---:|
| all-job makespan | 11.825x |
| mean flow | 11.758x |
| p90 flow | 11.814x |

All SOTA-style replay baselines select profile 13 on q00 under the same cache,
so q00 is a tie against SOTA-style policies and a large improvement over
Scheduleurm legacy. The four-quadrant and full five-taskset matrices remain
Pareto-undominated after this promotion.

## Active Taskset Update

The active q00 taskset now requires profiles 1-14:

```text
profiles 1-13: usable measured service curve
profile 14: measured local capacity boundary
```

The older profile 16 boundary remains in the cache as historical evidence but
is no longer the q00 closure used by `simulation/tasksets.py`.
