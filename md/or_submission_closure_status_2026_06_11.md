# OR Submission Closure Status

Date: 2026-06-11, Asia/Shanghai.

This note is the current reviewer-facing status after rerunning the production
coverage, production load, oracle-trace boundary, and Lean artifact checks.

## Closed Theorem-Facing Population

The theorem population is the Scheduleurm-controlled
`completed_active_production` view, not the raw scheduler history and not the
attempted-only stream.

Current Module51 status:

```text
global_theorem_closed = true
completed_active_production strict records = 3419
strict mapped = 3419
representative mapped = 0
unmapped = 0
measurement_required = 0
mapped_fraction = 1.0
```

Current Module49 mapped-slice capacity remains positive:

```text
strict mapped_capacity_usable_for_theorem = true
strict mapped delta = 0.00000912324072761479
representative mapped delta = 0.00000912324072761479
```

Module49 raw-window global coverage is intentionally false:

```text
raw 30-day records = 6933
strict raw mapped = 4803
strict raw unmapped = 2130
representative raw mapped = 6012
representative raw unmapped = 921
```

This is not a contradiction.  Raw history includes cancelled, forgotten,
benchmark, external auto-adopted, attempted-only, and other non-theorem
population rows.  It must not be written as a global stability theorem
population.

## Oracle Bridge Status

Module100 adds a service-map theorem oracle bridge over the same
completed-active production population:

```text
status = SERVICE_MAP_THEOREM_ORACLE_PASS
completed_active_record_count = 3419
alpha0 = 0.0
alpha1 = 0.0
usable_for_service_map_oracle_bridge = true
usable_for_live_scheduler_oracle_trace = false
```

This is a robust MaxWeight lower-service trace constructed from the measured
production service map.  It proves the theorem bridge and alpha0/alpha1 audit
close on the measured lower-service objects.

It is not a live dispatch trace.  The live scheduler trace line remains open
until a real candidate-family trace is captured from the scheduler and every
candidate is enriched with:

```text
queue_vector
lower_service
penalty_units
score_semantics = robust_maxweight_lower_service
```

Current live-trace modules are therefore correctly negative:

```text
Module50 scheduler trace audit: NO_TRACE
Module52 theorem oracle trace bridge: NO_TRACE
Module55 lower-service enrichment: NO_TRACE
```

## Lean Artifact

The consolidated Lean upload file was rechecked after the production artifact
updates:

```text
proof/ScheduleurmUpload.lean sha256 =
af79be4416e4c4add0fe41663fc0927af7058fe04412908b6688a8409227f01b

proof git HEAD =
23b101432067cc005512f7667810ec03b8cffb77

lake build Scheduleurm = pass
lake env lean ScheduleurmUpload.lean = pass
sorry/admit/axiom grep = no matches
```

## Submission Boundary

Safe to claim:

```text
The current completed-active Scheduleurm-controlled production population has
strict measured-bucket coverage, positive mapped-slice capacity slack, and a
service-map robust MaxWeight lower-service oracle bridge with alpha0=alpha1=0.
```

Do not claim:

```text
raw_history_all is closed;
attempted_production is closed;
future rolling queue rows are already closed;
the live scheduler dispatch implementation has already emitted a theorem-grade
lower-service oracle trace.
```

Primary current artifacts:

```text
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module100_production_theorem_oracle_bridge.json
md/experiment_artifacts/module100_production_theorem_oracle_trace.jsonl
md/experiment_artifacts/module50_scheduler_oracle_trace_status.json
md/experiment_artifacts/module52_theorem_oracle_trace_bridge.json
md/experiment_artifacts/module55_oracle_trace_enrichment_status.json
md/lean_verification_submission.md
```
