# Module 2 Sweetspot Clean Flow-Time Validation

Date: 2026-06-03

This log records the first passing live validation for the optional
`sweetspot_v1` placement module.  It is deliberately scoped to flow-time /
per-job service-rate performance.  It does not claim aggregate capacity
improvement.

## Run

- Run id: `module2_logged_clean_flow_20260603_0005_size16384`
- Run directory:
  `/home/erzhu419/.claude/scheduler/experiments/runs/module2_logged_clean_flow_20260603_0005_size16384`
- Node: `jtl110gpu2`
- Workload: 6 identical JAX GPU progress benchmarks
- Benchmark size: `16384`
- Benchmark steps: `1800`
- Measurement window: `120s` after warmup
- Performance objective: `flow_time`
- Hard-rule mode: `clean_bench`
- Watcher preflight: inactive

## Clean A/B Setup

Legacy baseline:

- Algorithm: `legacy`
- Scheduler hard-rule override: `clean_bench`
- Placement: 6 active tasks, 3 per GPU
- Intent: measure clean over-packed legacy placement without one-third,
  startup-reserve, util-saturation, max-task, thread-pressure, or margin gates.

Sweetspot candidate:

- Algorithm: `sweetspot_v1`
- Scheduler hard-rule override: `clean_bench`
- Algorithm gate: `SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU=2`
- Placement: 4 active tasks, 2 per GPU, 2 queued by the algorithm gate
- Intent: validate the algorithmic co-location cap, not legacy hard rules.

## Result

| Metric | Legacy | Sweetspot | Ratio |
| --- | ---: | ---: | ---: |
| Mean active service rate | 1.371382 step/s | 2.018975 step/s | 1.472219 |
| Aggregate active service rate | 8.228292 step/s | 8.075900 step/s | 0.981480 |
| Mean completion-time proxy | 1312.545 s | 1188.722 s | 1.104164 |

Validation status:

- Placement pass: yes. Legacy ran 3/GPU; sweetspot held 2/GPU and queued the
  remaining tasks by `algorithm:sweetspot_v1: task_count 2/2`.
- Measurement pass: yes. Legacy had 6/6 rates; sweetspot had 4/4 active rates.
- Flow-time performance pass: yes. Mean completion-time proxy improved by
  10.4%, exceeding the 1.05 threshold.
- Aggregate objective pass: no. Aggregate service rate was 0.981x, so this run
  must not be cited as aggregate capacity improvement.
- Eviction/rollback: none in the passing run.

## Evidence Artifacts

- `reports/verdict.json`: final pass/fail and metric ratios.
- `reports/events.jsonl`: phase events, watcher status, warmup samples,
  measurement samples, summaries, verdict.
- `reports/timeline.jsonl`: normalized per-sample phase summaries.
- `reports/per_task_timeline.jsonl`: per-task status, GPU, rate, progress,
  placement audit, hard-rule mode, block/eviction fields.
- `raw/*.stdout`, `raw/*.stderr`, `raw/*.json`: scheduler command outputs and
  queue snapshots.

## Notes

Earlier runs without watcher isolation were invalid because the systemd watcher
used the installed scheduler copy and performed `gpu_one_third` rollback during
the sweetspot phase.  The passing run required watcher inactive preflight and
records that preflight in `events.jsonl`.

This module is now validated for flow-time / per-job service-rate improvement.
Aggregate capacity and full service-curve calibration remain separate
experimental modules.
