# Q01 GPU-Heavy Profiles 4-8 Validation

Date: 2026-06-04

This module fills the `q01_gpu_bound_compute` measurement gap from
`simulation/tasksets.py`. It measures the same JAX matmul benchmark as module 6,
but extends the co-location curve from profiles 1-3 to profiles 4-8.

The runner cancels tasks after a stable progress window. It does not wait for
the benchmark jobs to run to natural completion.

## Command

```bash
python3 -m algorithm.experiments.service_curve_validation \
  --run-id module22_q01_gpu_heavy_jax8192_profiles4_8_20260604_001 \
  --node jtl110gpu2 \
  --gpus 0,1 \
  --profiles 4,5,6,7,8 \
  --cwd /home/erzhu419/scheduleurm_bench_cwd \
  --remote-script /tmp/scheduleurm_gpu_progress_benchmark.py \
  --python-bin /home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python \
  --steps 2400 \
  --size 8192 \
  --vram-mb 800 \
  --ram-mb 4096 \
  --cpu 1 \
  --mem-fraction 0.20 \
  --warmup-timeout-s 360 \
  --measure-s 90 \
  --poll-s 30 \
  --hard-rule-mode clean_bench \
  --expected-sweetspot-count 0 \
  --total-jobs-for-proxy 24 \
  --proxy-job-counts 24,48,96 \
  --min-two-vs-three-gain 0.0
```

## Result

Run directory:

```text
/home/erzhu419/.claude/scheduler/experiments/runs/module22_q01_gpu_heavy_jax8192_profiles4_8_20260604_001
```

Validation status: pass.

| Profile | Running | With rate | Median aggregate step/s | Placement | Blocked |
|---:|---:|---:|---:|:---|---:|
| 4/GPU | 8 | 8 | 70.107500 | ok | 0 |
| 5/GPU | 10 | 10 | 70.001248 | ok | 0 |
| 6/GPU | 12 | 12 | 70.208424 | ok | 0 |
| 7/GPU | 14 | 14 | 70.943463 | ok | 0 |
| 8/GPU | 16 | 16 | 71.264482 | ok | 0 |

No capacity boundary was observed for 4-8/GPU under this synthetic GPU-heavy
benchmark. The aggregate service curve is nearly flat after 4/GPU, while
per-task service drops as co-location increases.

## Interpretation

Together with module 6 profiles 1-3, the q01 curve now has exact measured
profiles 1-8. For small batches, 1/GPU remains best by completion time because
per-task service is much higher. For larger batches, high co-location can reduce
makespan by increasing active slots while keeping aggregate service flat.

This supports the algorithmic need for taskset-aware profile selection rather
than a fixed global cap.

After adding profiles 4-8 to the default cache, q01 standalone replay selects
8/GPU against the legacy 3/GPU cap:

| Taskset | Objective | Legacy | Calibrated | Ratio |
|---|---|---:|---:|---:|
| `q01_gpu_bound_compute` | all-task makespan | 3/GPU | 8/GPU | 1.096x |
| `q01_gpu_bound_compute` | mean flow | 3/GPU | 8/GPU | 0.940x |

This is not a contradiction. The current calibrated replay policy optimizes
all-task makespan, not mean flow. The result should be reported under the
makespan objective unless the policy objective is changed.
