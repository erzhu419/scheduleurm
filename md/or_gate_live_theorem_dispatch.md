# OR Gate: Launched Live Theorem Dispatch

Date: 2026-06-11, Asia/Shanghai.

## Summary

| Quantity | Value |
|---|---:|
| status | `PASS` |
| controlled task count | 6 |
| dispatch pass count | 2 |
| trace slots | 7 |
| theorem slots | 7 |
| candidate rows | 13 |
| scheduler dispatch return code | 0 |
| wait return code | 0 |
| oracle alpha0 | 0 |
| oracle alpha1 | 0 |
| realization status | `LIVE_COMPLETION_REALIZATION_PASS` |

## Artifact Set

- report: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003.json`
- trace: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_trace.jsonl`
- realization bridge: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_realization.json`
- markdown run summary: `md/live_theorem_dispatch_bounded_portfolio_20260611_003.md`
- implementation status: `md/live_theorem_dispatch_status_2026_06_11.md`

## Scope

This gate is a launched, queue-mutating ScheduleurmBench validation. The tasks
were submitted through `scheduler.py submit`, dispatched through
`scheduler.py dispatch --algorithm theorem_maxweight_v1 --task-id ...`, launched
as real processes, waited to completion, and matched back to scheduler queue
records by the live realization bridge.

The candidate family was intentionally bounded to avoid interfering with
existing user jobs:

```text
allowed_nodes = ["local", "node007-direct"]
max_gpu_util_pct = 20
require_node = ""
preferred_node = ""
```

The trace contains seven theorem slots for six controlled tasks because one
task was initially selected for `node007-direct|gpu=3`; launch failed on a
missing remote cwd, and the experiment driver's second pass retried the same
task under the scheduler's launch-failed soft block, selecting `local|gpu=0`.

## Claim Boundary

This closes the earlier dry-run-to-live gap for bounded q01/q11 controlled
workloads: the theorem policy can produce robust lower-service candidate-family
slots, launch real tasks, and bridge selected actions to completed records.

It should not be stated as a production-wide theorem trace. It is not evidence
that every candidate node is deployment-ready, nor that all future scheduler
dispatches automatically satisfy the theorem contract. It is a bounded live
candidate-family validation with one observed operational fallback retry.
