# Live Theorem Dispatch Status 2026-06-11

## What Closed

The first controlled live theorem-dispatch smoke run passed.

Artifacts:

- report: `md/experiment_artifacts/live_theorem_dispatch_smoke_20260611_001.json`
- trace: `md/experiment_artifacts/live_theorem_dispatch_smoke_20260611_001_trace.jsonl`
- realization bridge: `md/experiment_artifacts/live_theorem_dispatch_smoke_20260611_001_realization.json`
- markdown summary: `md/live_theorem_dispatch_smoke_20260611_001.md`

Observed certificate:

- `dispatch_returncode = 0`
- `wait_returncode = 0`
- `trace_slot_count = 2`
- `theorem_slot_count = 2`
- `candidate_count_total = 2`
- theorem bridge `alpha0 = 0`, `alpha1 = 0`
- realization status `LIVE_COMPLETION_REALIZATION_PASS`

This is the first non-dry-run evidence that `scheduler.py dispatch
--algorithm theorem_maxweight_v1` can launch real tasks, emit
`robust_maxweight_lower_service` candidate-family trace slots, and bridge those
slots to completed scheduler task records.

## Exact Scope

This run is a controlled local smoke test:

- both tasks were submitted as `ScheduleurmBench`;
- both were hard-pinned with `require_node = local`;
- the workload labels were one q01-like `gpu_heavy_jax_matmul` task and one
  q11-like `hybrid_rl_resac_ant` task;
- the commands used `algorithm/experiments/gpu_progress_benchmark.py` as a short
  live workload carrier.

It should not be described as a full global q01/q11 production experiment.
Because of the hard pin, each trace slot had only one feasible selected local
candidate. The result closes the dry-run-to-live realization gap for the
theorem hook, not the full global candidate-family claim.

## Attempted But Not Claimed

A second run attempted to remove the hard pin and use `preferred_node = local`
so the scheduler would construct a broader candidate family while likely still
choosing local. That run did not reach trace emission within the smoke window:
without a hard pin, the current scheduler path enters full-node probe/staging
before launch. The run was terminated before launch and its queued tasks were
cancelled:

- `t10293`: cancelled before launch
- `t10294`: cancelled before launch

No theorem or performance claim is made from that second run.

## Bounded Candidate Run

The bounded-candidate live theorem run passed after exposing the existing
`allowed_nodes` candidate-family constraint through the scheduler CLI.

Artifacts:

- report: `md/experiment_artifacts/live_theorem_dispatch_bounded_candidate_20260611_001.json`
- trace: `md/experiment_artifacts/live_theorem_dispatch_bounded_candidate_20260611_001_trace.jsonl`
- realization bridge: `md/experiment_artifacts/live_theorem_dispatch_bounded_candidate_20260611_001_realization.json`
- markdown summary: `md/live_theorem_dispatch_bounded_candidate_20260611_001.md`

Observed certificate:

- `require_node = ""`
- `preferred_node = ""`
- `allowed_nodes = ["local", "node007-direct"]`
- `max_gpu_util_pct = 20`
- `dispatch_returncode = 0`
- `wait_returncode = 0`
- `trace_slot_count = 2`
- `theorem_slot_count = 2`
- `candidate_count_total = 4`
- theorem bridge `alpha0 = 0`, `alpha1 = 0`
- realization status `LIVE_COMPLETION_REALIZATION_PASS`

The emitted slots were fallback-phase theorem slots, not hard-pin slots.  Each
slot contained two theorem-certified candidates:

- `node=local|gpu=0`
- `node=node007-direct|gpu=3`

The selected action was `node=local|gpu=0` for both controlled q01/q11 tasks.
This is now a real live candidate-family trace with action choice made by
`theorem_maxweight_v1` over a bounded safe node set. It still should not be
overclaimed as a full production-wide live trace: the population is the
controlled `ScheduleurmBench` q01/q11 pair, and the candidate family was
experimentally bounded to avoid interfering with existing user jobs.

## Bounded Portfolio Run

The reproducible longer bounded-candidate portfolio run also passed after
adding the experimental `dispatch --task-id` one-shot filter and multi-pass
retry loop to the experiment driver.

Artifacts:

- report: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003.json`
- trace: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_trace.jsonl`
- realization bridge: `md/experiment_artifacts/live_theorem_dispatch_bounded_portfolio_20260611_003_realization.json`
- markdown summary: `md/live_theorem_dispatch_bounded_portfolio_20260611_003.md`

Observed certificate:

- controlled task count: `6`
- dispatch pass count: `2`
- `require_node = ""`
- `preferred_node = ""`
- `allowed_nodes = ["local", "node007-direct"]`
- `max_gpu_util_pct = 20`
- `dispatch_returncode = 0`
- `wait_returncode = 0`
- `trace_slot_count = 7`
- `theorem_slot_count = 7`
- `candidate_count_total = 13`
- theorem bridge `alpha0 = 0`, `alpha1 = 0`
- realization status `LIVE_COMPLETION_REALIZATION_PASS`

The trace has seven slots for six tasks because one task was first selected for
`node007-direct|gpu=3`; launch then failed on a missing remote cwd, so the
second pass retried the same task under the scheduler's launch-failed soft
block and selected `local|gpu=0`. This is useful evidence, not a failure: the
live trace now covers both theorem action selection and the scheduler's
operational fallback boundary. It should be described as a bounded live
candidate-family validation with one launch-fallback retry, not as a direct
proof that every candidate node is already deployment-ready.

## Next Required Work

To upgrade from bounded-candidate q01/q11 smoke to the reviewer-facing live
theorem trace, run longer q01/q11/portfolio controlled tasks with:

- actual queue mutation and process launch;
- `SCHEDULEURM_ORACLE_TRACE_PATH` set;
- `SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block`;
- candidate family size greater than one;
- completed realization bridge;
- no unbounded remote staging;
- enough slots to estimate live oracle-gap and completion/JCT agreement beyond
  the two-slot smoke.
