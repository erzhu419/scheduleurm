# Module55 Oracle Trace Lower-Service Enrichment

Date: 2026-06-08

Module50 records live scheduler candidate families.  Module52 only accepts
theorem-grade robust MaxWeight lower-service trace slots.  Module55 bridges
those two artifacts: it takes a scheduler candidate-family trace, a measured
lower-service lookup, and a queue vector, then rewrites each trace slot into the
Module52 schema or refuses the certificate with explicit blockers.

## Artifacts

```text
algorithm/experiments/oracle_trace_enrichment.py
md/experiment_artifacts/module55_empty_service_lookup.json
md/experiment_artifacts/module55_empty_queue_vector.json
md/experiment_artifacts/module55_oracle_trace_enrichment_status.json
md/experiment_artifacts/module55_oracle_trace_enrichment_status.md
```

## Accepted Inputs

The candidate trace is the JSONL emitted by Scheduleurm when one of these
environment variables is set:

```text
SCHEDULEURM_ORACLE_AUDIT_LOG=/path/to/oracle_trace.jsonl
SCHEDULEURM_ORACLE_TRACE_PATH=/path/to/oracle_trace.jsonl
```

The lower-service lookup may be a JSON list or an object containing `rows`,
`records`, `actions`, `candidate_services`, or `service_rows`.  Each service row
can match a candidate by:

```text
action_id
candidate_bucket
class_key
regime_key
```

and must carry:

```text
lower_service or lower_service_vector or service_vector
optional penalty_units
optional source
```

The queue vector must be nonempty.  Missing queue state is a theorem blocker
because the oracle gap is:

```text
oracle_gap(k) <= alpha0 + alpha1 * ||Q(k)||_1
```

## Current Status

Current production status:

```text
status = NO_TRACE
input = /home/erzhu419/.claude/scheduler/oracle_trace.jsonl
usable_for_theorem = false
blocker = trace_file_does_not_exist
```

The regression test validates the non-production bridge on a two-candidate
slot:

```text
partial lower-service lookup -> ENRICHMENT_BLOCKED
complete bucket lookup       -> ENRICHED_THEOREM_PASS
oracle_gap                   -> 3.0
alpha1                       -> 0.3
```

## Interpretation

This closes the software path from live candidate traces to theorem constants
`alpha0, alpha1`.  It does not claim a production oracle certificate yet.  A
theorem-grade production pass still requires a real trace, a lower-service row
for every candidate in every traced slot, and the correct queue vector for the
decision state.
