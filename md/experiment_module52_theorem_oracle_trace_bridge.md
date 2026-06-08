# Module52 Theorem Oracle Trace Bridge

Date: 2026-06-08

Module50 records live scheduler candidate families and audits scheduler-sort-key
optimality.  Module52 is stricter: it only converts trace slots into
theorem-side approximate-oracle certificates when the trace carries robust
MaxWeight lower-service semantics.

## Artifacts

```text
algorithm/experiments/theorem_oracle_trace_bridge.py
md/experiment_artifacts/module52_theorem_oracle_trace_bridge.json
md/experiment_artifacts/module52_theorem_oracle_trace_bridge.md
```

## Accepted Trace Schema

Each accepted slot must contain:

```text
score_semantics = robust_maxweight_lower_service
queue_vector
candidate action_id
candidate lower_service
candidate penalty_units
exactly one selected candidate
```

The bridge then emits the `oracle_audit.py` slot format and estimates:

```text
oracle_gap(k) <= alpha0 + alpha1 ||Q(k)||_1
```

## Current Status

Current production trace status:

```text
status = NO_TRACE
trace_slot_count = 0
converted_slot_count = 0
usable_for_theorem = false
```

Reason:

```text
~/.claude/scheduler/oracle_trace.jsonl does not exist yet
```

This is an honest blocker, not a theorem failure.  The implementation path is
now closed: once real decisions are traced with lower-service vectors, the
bridge can directly produce `alpha0, alpha1`.

## Why This Matters

A scheduler-sort-key trace can pass Module50 while still failing Module52.
That separation is intentional:

```text
Module50: implementation chose best candidate under its own recorded sort key.
Module52: chosen candidate satisfies approximate robust MaxWeight lower-service audit.
```

Only Module52 output may be used as the theorem-condition `alpha0, alpha1`
certificate.
