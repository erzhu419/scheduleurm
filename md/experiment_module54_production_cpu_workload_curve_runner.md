# Module54 Production CPU Workload Curve Runner

Date: 2026-06-08

Module53 identified `cpu_sumo_transit_eval_or_control` as the dominant
production coverage blocker.  Module54 adds the concrete runner for that
blocker: a CPU-only service-curve probe that submits real production workload
commands through Scheduleurm, wraps them with a progress counter, waits only
until stable progress is observed, and then cancels the profile batch.

## Artifacts

```text
algorithm/experiments/production_cpu_workload_curve.py
md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.json
md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.md
```

## Current Dry-Run Plan

```text
run_id = module54_freqduet_cpu_c17_32_plan
sub_bucket = freqduet_cpu_ablation|c_17_32
node = local
profiles = 1, 2, 4, 8
task_count = 15
cpu/task = 24
ram_mb/task = 65536
total_units/task = 20 episodes
theorem_status = plan_only_not_measured
```

The rendered workload is the FreqDuet ablation command:

```text
python scripts/run_freqduet_ablation.py
  --configs F_freqduet_terminal_main_hiro
  --seeds <seed>
  --episodes <total_units>
  --workers 24
  --worker-threads 1
  --logs-dir <output_root>/<run_name>
```

Each submitted task requests:

```text
vram_mb = 0
require_node = local
hard_rule_mode = clean_bench
allow_duplicate = true
allow_no_ckpt = true
allow_no_resume = true
```

## Command Surface

Dry-run review:

```bash
PYTHONPATH=. python3 -m algorithm.experiments.production_cpu_workload_curve \
  --dry-run \
  --run-id module54_freqduet_cpu_c17_32_plan \
  --sub-bucket 'freqduet_cpu_ablation|c_17_32' \
  --node local \
  --profiles 1,2,4,8 \
  --cwd /home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet \
  --cmd-template 'set -euo pipefail; python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro --seeds {seed} --episodes {total_units} --workers {cpu_cores} --worker-threads 1 --logs-dir {output_root}/{run_name}' \
  --output-root scheduleurm_production_cpu_curve_runs \
  --seed-base 700000 \
  --total-units 20 \
  --progress-unit episode \
  --cpu 24 \
  --ram-mb 65536 \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/production_cpu_workload_curve \
  --plan-output md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.json \
  --plan-markdown-output md/experiment_artifacts/module54_freqduet_cpu_c17_32_plan.md
```

Actual measurement uses the same command without `--dry-run`.  The runner
copies the progress wrapper to the target node, submits one profile at a time,
measures a stable progress window, writes `service_curve_verdict.json`, and
cancels all submitted tasks after measurement.

## Interpretation

This attacks the production coverage blocker without changing the theorem or
weakening the claim.  It is not yet a service certificate: the current artifact
is a reviewed submission plan only.  The theorem condition becomes eligible for
recomputation only after this runner produces measured profile summaries and
those summaries are loaded into the measured service cache.
