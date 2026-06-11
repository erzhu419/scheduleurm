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
input = md/experiment_artifacts/module101_live_scheduler_oracle_trace_enriched.jsonl
status = THEOREM_ORACLE_PASS
trace_slot_count = 2
converted_slot_count = 2
alpha0 = 0.0
alpha1 = 0.0
usable_for_theorem = true
```

Scope:

```text
one emitted live scheduler candidate-family trace after Module55 lower-service enrichment
```

The previous `NO_TRACE` blocker is closed for this captured trace.  This does
not make untraced future dispatches theorem-grade automatically; every future
online-oracle claim still needs the same Module50 -> Module55 -> Module52
artifact chain.

Module55 now implements the enrichment feeder for this strict gate:

```text
scheduler candidate-family trace
+ measured lower-service lookup
+ decision queue vector
-> robust_maxweight_lower_service trace
```

This does not relax Module52.  If any candidate lacks lower-service semantics,
or if the queue vector is missing, the oracle certificate remains blocked.

## Why This Matters

A scheduler-sort-key trace can pass Module50 while still failing Module52.
That separation is intentional:

```text
Module50: implementation chose best candidate under its own recorded sort key.
Module52: chosen candidate satisfies approximate robust MaxWeight lower-service audit.
```

Only Module52 output may be used as the theorem-condition `alpha0, alpha1`
certificate.
