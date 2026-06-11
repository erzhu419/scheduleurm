# module96_resac_review5_jax_train_fabric_completed_history

Date: 2026-06-11

Strict production-fabric completed-history lower-service certificate for RE-SAC review5 JAX GPU training commands. The progress unit is one completed training command.

```text
workload_key = resac_review5_jax_train_fabric_completed_history
record_count = 151
completed_record_count = 151
profile_domain = [1]
burst_span_days = 5.930530836
lower_service = 0.000294692876 command/s
arrival_lambda_30d = 0.000058256173 command/s
service_to_arrival_ratio = 5.058569
theorem_status = strict_production_fabric_completed_history_lower_service
```

## Scope

This is a fabric-level service certificate over the observed production GPU pool. It is not a single-GPU throughput curve. The lower-service point is computed as completed commands divided by the elapsed time from first submission to last completion in the observed burst, so queueing, contention, and adoption overhead are included rather than subtracted.

## Artifacts

```text
md/experiment_artifacts/module96_resac_review5_jax_train_fabric_completed_history.json
md/experiment_artifacts/module96_resac_review5_jax_train_fabric_completed_history.md
md/experiment_artifacts/module96_resac_review5_jax_train_fabric_completed_history_reports/profile_1_per_resource_summary.json
```
