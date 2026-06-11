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
completed_active_production strict records = 3437
strict mapped = 3437
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
raw 30-day records = 6957
strict raw mapped = 4825
strict raw unmapped = 2132
representative raw mapped = 6034
representative raw unmapped = 923
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
completed_active_record_count = 3437
alpha0 = 0.0
alpha1 = 0.0
usable_for_service_map_oracle_bridge = true
usable_for_live_scheduler_oracle_trace = false
```

This is a robust MaxWeight lower-service trace constructed from the measured
production service map.  It proves the theorem bridge and alpha0/alpha1 audit
close on the measured lower-service objects.

It is not itself a live dispatch trace.  Module101 separately closes one emitted
live scheduler candidate-family trace by enriching every observed candidate
with:

```text
queue_vector
lower_service
penalty_units
score_semantics = robust_maxweight_lower_service
```

Current live-trace modules now pass on that emitted trace:

```text
Module50 scheduler trace audit: SCHEDULER_SCORE_PASS
Module55 lower-service enrichment: ENRICHED_THEOREM_PASS
Module52 theorem oracle trace bridge: THEOREM_ORACLE_PASS
Module101 live scheduler oracle closure: LIVE_SCHEDULER_THEOREM_ORACLE_PASS
Module101 trace slots = 2
Module101 candidate count = 2
Module101 alpha0 = 0.0
Module101 alpha1 = 0.0
```

The Module101 scope is intentionally narrow: one emitted live scheduler trace
over the local CPU control-plane candidate family.  It removes the previous
`NO_TRACE` blocker and proves the pipeline is executable, but it is not a
universal certificate for every future dispatch or every GPU candidate family.

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
every future scheduler dispatch is automatically theorem-grade without rerunning
the trace/enrichment/audit pipeline.
```

Primary current artifacts:

```text
md/experiment_artifacts/module49_production_load_strict.json
md/experiment_artifacts/module49_production_load_representative.json
md/experiment_artifacts/module51_production_coverage_drilldown.json
md/experiment_artifacts/module100_production_theorem_oracle_bridge.json
md/experiment_artifacts/module100_production_theorem_oracle_trace.jsonl
md/experiment_artifacts/module101_live_scheduler_oracle_closure.json
md/experiment_artifacts/module101_live_scheduler_oracle_trace_raw.jsonl
md/experiment_artifacts/module101_live_scheduler_oracle_trace_enriched.jsonl
md/experiment_artifacts/module50_scheduler_oracle_trace_status.json
md/experiment_artifacts/module52_theorem_oracle_trace_bridge.json
md/experiment_artifacts/module55_oracle_trace_enrichment_status.json
md/lean_verification_submission.md
```
