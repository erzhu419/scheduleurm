# Module32 Replay-To-Live Sanity Validation

Date: 2026-06-05

This module adds the artifact needed to connect offline replay to small real
end-to-end runs. It does not require long production jobs to run to natural
completion; it compares small live completion/JCT observations against explicit
task-list replay predictions.

## Artifact

```text
algorithm/experiments/live_validation.py
skill/tests/test_experiment_calibration.py
```

## Replay Input

Generate an explicit task-list replay report:

```bash
python3 -m simulation.trace_benchmark_cli \
  --taskset q01_gpu_bound_compute \
  --arrival-mode static \
  --seed 42 \
  --replay-seed 7 \
  --report-out /tmp/q01_replay_report.json
```

The live validation tool reads the calibrated candidate row by default, or a
specific `--policy` if provided.

## Live Input Schema

The live report is intentionally simple:

```json
{
  "run_id": "q01_live_small_001",
  "jobs": [
    {
      "job_id": "j0",
      "arrival_s": 0.0,
      "completion_s": 120.0
    }
  ]
}
```

Rows may use `flow_s` directly instead of `arrival_s`/`completion_s`. Censored
rows are allowed for engineering diagnostics but make the report unusable for
theorem-facing live sanity:

```json
{"job_id": "j1", "flow_s": 95.0, "censored": true}
```

## Command

```bash
python3 -m algorithm.experiments.live_validation compare \
  --replay-report /tmp/q01_replay_report.json \
  --live-report /tmp/q01_live_report.json \
  --max-relative-error 0.20 \
  --output /tmp/q01_live_validation.json
```

The report includes:

```text
predicted makespan / mean-flow / p90-flow
observed makespan / mean-flow / p90-flow
absolute and relative error for each metric
completed job count match
censored job count
usable_for_live_sanity
```

## Required Live Runs

Before claiming replay-backed system validation, run at least:

```text
q01 small live sanity
q11 small live sanity
hybrid portfolio small live sanity
```

The purpose is not to rerun every large replay trace to completion. The purpose
is to show that measured service curves and explicit task-list replay predict
small real end-to-end completion/JCT within a stated error tolerance.
