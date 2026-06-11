# Module50 Scheduler Candidate-Set Oracle Trace

Date: 2026-06-08

This module adds the missing logging surface for the live scheduler oracle
audit.  Before this change, Scheduleurm task history recorded only the selected
GPU placement audit:

```text
placement_algorithm_audit = selected action only
```

That is not enough to estimate the approximate-oracle constants
`alpha0, alpha1`, because the theorem needs the selected action's gap against
the full candidate family at the same decision state.

## Artifacts

```text
algorithm/oracle_trace.py
algorithm/experiments/scheduler_oracle_trace_audit.py
skill/scheduler.py
skill/tests/test_algorithm_policy.py
md/experiment_artifacts/module50_scheduler_oracle_trace_status.json
md/experiment_artifacts/module50_scheduler_oracle_trace_status.md
```

## Scheduler Integration

The trace hook is disabled by default.  It becomes active only when one of these
environment variables is set:

```text
SCHEDULEURM_ORACLE_AUDIT_LOG=/path/to/oracle_trace.jsonl
SCHEDULEURM_ORACLE_TRACE_PATH=/path/to/oracle_trace.jsonl
```

When enabled, `pick_placement` writes one JSONL slot for the actual candidate
family used by the selected phase:

```text
require_node
resume_preferred
preferred_node
fallback
```

This distinction matters.  A required-node decision should not be audited
against GPUs on other nodes that were infeasible under the hard pin; a preferred
or resume-locality decision similarly has a smaller operational candidate
family than the global fallback search.

Each slot records:

```text
task id / project / signature / priority
algorithm name
phase
hard constraints
candidate_count
selected_action_id
per-candidate node/gpu, sort key, primary numeric score
per-candidate finite bucket / class / regime audit when available
```

## Audit CLI

After collecting a trace:

```text
PYTHONPATH=. python3 -m algorithm.experiments.scheduler_oracle_trace_audit audit \
  --input ~/.claude/scheduler/oracle_trace.jsonl \
  --output md/experiment_artifacts/module50_scheduler_oracle_trace_status.json \
  --markdown-output md/experiment_artifacts/module50_scheduler_oracle_trace_status.md
```

The CLI checks whether the selected candidate is best under the recorded
scheduler sort key.  It intentionally separates two statuses:

```text
usable_for_scheduler_score_audit
usable_for_theorem
```

A scheduler-score PASS is useful engineering evidence that the implementation
chooses the best candidate it generated.  It is not automatically a robust
MaxWeight theorem certificate.

## Current Status

Current artifact:

```text
input = md/experiment_artifacts/module101_live_scheduler_oracle_trace_raw.jsonl
status = SCHEDULER_SCORE_PASS
trace_slot_count = 2
audited_slot_count = 2
usable_for_scheduler_score_audit = true
usable_for_theorem = false
```

Theorem blocker:

```text
trace uses scheduler sort-key semantics, not robust MaxWeight lower-service semantics
```

Captured slots:

```text
slot:t9981:1781145424.92721:1781145677613 -> node=local|cpu
slot:t9983:1781145424.92721:1781145827811 -> node=local|cpu
scheduler_score_gap = 0
```

The regression test also validates the mechanism on a synthetic two-GPU
decision, so both the synthetic and emitted-live paths now exercise the trace
audit code.

## Remaining Theorem Condition

To turn this into the theorem-side approximate-oracle constants, the trace must
carry robust MaxWeight lower-service semantics:

```text
candidate action lower_service vectors
queue_vector at the decision state
penalty_units
chosen_action_id
best score over the same finite candidate family
```

Then `algorithm/experiments/oracle_audit.py` can estimate:

```text
oracle_gap(k) <= alpha0 + alpha1 * ||Q(k)||_1
```

Raw Module50 output is intentionally not the theorem-side certificate.  The
important correction is that selected-only history can no longer be mistaken for
a full candidate-set certificate, and scheduler-sort-key optimality is kept
separate from robust MaxWeight lower-service optimality.

Module55 now implements the reconstruction path:

```text
scheduler candidate-family trace
+ measured lower-service lookup
+ decision queue vector
-> robust_maxweight_lower_service trace
-> Module52 alpha0/alpha1 audit
```

For the captured Module101 trace this path now closes through Module55 and
Module52 with `alpha0 = alpha1 = 0`.  Future dispatches still need their own
trace, measured lower-service rows, and queue vector before they are used as
theorem-grade online oracle evidence.
