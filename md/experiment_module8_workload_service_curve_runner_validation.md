# Module 8: Real-Workload Service-Curve Runner

This module adds the runner needed to validate the RL observation directly:
some RL jobs can run at 4-5/GPU with nearly unchanged per-task wall time because
their GPU usage is bursty rather than continuously compute-bound.

The module does not change scheduler placement behavior.  It submits tasks
through the existing Scheduleurm CLI, pins them to fixed GPU profiles, dispatches
with the selected hard-rule mode, measures task-native progress rate, and
cancels the jobs after the measurement window.

## Artifact

`algorithm/experiments/workload_service_curve_validation.py`

The runner accepts a command template with explicit placeholders:

```text
{run_id}
{phase}
{profile}
{count_per_gpu}
{index}
{gpu_idx}
{seed}
{max_iters}
{output_root}
{run_name}
{node}
```

Literal shell braces must be escaped as `{{` and `}}`, because Python string
formatting is used deliberately.  Unknown placeholders fail before submission,
so a typo cannot silently create duplicate task identities.

## Example Shape

```bash
python3 -m algorithm.experiments.workload_service_curve_validation \
  --run-id rl_curve_smoke \
  --node jtl110gpu2 \
  --gpus 0 \
  --profiles 1,2,3,4,5 \
  --cwd /home/erzhu419/mine_code/RE-SAC \
  --project ScheduleurmBench \
  --signature-prefix ScheduleurmBench/workload_service_curve/resac \
  --max-iters 80 \
  --total-units 80 \
  --cmd-template 'PYTHONPATH=/home/erzhu419/mine_code/RE-SAC:$PYTHONPATH XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 /home/erzhu419/.conda/envs/resac-jax/bin/python -u -m jax_experiments.train --algo sac --ensemble_size 2 --env Walker2d-v2 --seed {seed} --max_iters {max_iters} --checkpoint_interval 100000 --save_root {output_root} --backend spring --device gpu --run_name {run_name}'
```

The command above is a template, not a completed validation result.  The actual
performance validation must be run on an agreed RL class and profile set, then
its `service_curve_verdict.json` and `service_curve.md` become the evidence.

## Functional Validation

Targeted checks passed:

```text
PASS workload template rejects unknown placeholder
PASS workload template renders deterministic seed
PASS workload template renders unique run name
PASS workload template passes max_iters and output root
PASS workload service curve can certify RL iter sweetspot
PASS progress parser reads RL iter and cmd total
PASS progress parser converts s/iter to iter/s
PASS sweetspot summary counts RL iter/s rates
PASS service curve verdict passes two-per-GPU flow sweetspot
```

Compile checks passed for the runner and tests.

No live RL tasks were launched for this module.  That is intentional: the next
module should use this runner to perform the real fixed-profile RL service-curve
experiment one profile at a time.
