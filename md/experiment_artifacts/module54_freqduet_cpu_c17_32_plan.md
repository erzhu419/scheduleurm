# Production CPU Workload Service-Curve Plan

```text
run_id = module54_freqduet_cpu_c17_32_plan
sub_bucket = freqduet_cpu_ablation|c_17_32
node = local
task_count = 15
progress_unit = episode
total_units = 20
theorem_status = plan_only_not_measured
```

| Profile | Tasks | CPU/task | RAM MB/task |
|---:|---:|---:|---:|
| 1 | 1 | 24 | 65536 |
| 2 | 2 | 24 | 65536 |
| 4 | 4 | 24 | 65536 |
| 8 | 8 | 24 | 65536 |

## First Commands

### `ScheduleurmBench/production_cpu_workload_curve/freqduet_cpu_ablation|c_17_32/module54_freqduet_cpu_c17_32_plan/profile_1_per_resource/0`

```bash
python3 -u /tmp/scheduleurm_progress_wrapper_pkg/algorithm/experiments/progress_wrapper.py --unit episode --total 20 -- bash -lc 'set -euo pipefail; python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro --seeds 700000 --episodes 20 --workers 24 --worker-threads 1 --logs-dir scheduleurm_production_cpu_curve_runs/module54_freqduet_cpu_c17_32_plan_freqduet_cpu_ablation_c_17_32_profile_1_per_resource_0'
```

### `ScheduleurmBench/production_cpu_workload_curve/freqduet_cpu_ablation|c_17_32/module54_freqduet_cpu_c17_32_plan/profile_2_per_resource/0`

```bash
python3 -u /tmp/scheduleurm_progress_wrapper_pkg/algorithm/experiments/progress_wrapper.py --unit episode --total 20 -- bash -lc 'set -euo pipefail; python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro --seeds 700000 --episodes 20 --workers 24 --worker-threads 1 --logs-dir scheduleurm_production_cpu_curve_runs/module54_freqduet_cpu_c17_32_plan_freqduet_cpu_ablation_c_17_32_profile_2_per_resource_0'
```

### `ScheduleurmBench/production_cpu_workload_curve/freqduet_cpu_ablation|c_17_32/module54_freqduet_cpu_c17_32_plan/profile_2_per_resource/1`

```bash
python3 -u /tmp/scheduleurm_progress_wrapper_pkg/algorithm/experiments/progress_wrapper.py --unit episode --total 20 -- bash -lc 'set -euo pipefail; python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro --seeds 700001 --episodes 20 --workers 24 --worker-threads 1 --logs-dir scheduleurm_production_cpu_curve_runs/module54_freqduet_cpu_c17_32_plan_freqduet_cpu_ablation_c_17_32_profile_2_per_resource_1'
```

## Interpretation

This is a dry-run submission plan.  It is not a service certificate until
the runner submits the tasks, observes stable progress, writes profile
summaries, and those summaries are loaded into the service cache.
