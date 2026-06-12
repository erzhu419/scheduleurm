# Service-Certified Admission Population

## Summary

| Population | Records | Strict admitted | Domain admitted | Closed |
|---|---:|---:|---:|---:|
| completed-active production | 3545 | 3540 | 3545 | false |
| attempted production | 5340 | 3919 | 5339 | false |
| active queue | 48 | 47 | 48 | false |

## Scope

the reviewer-facing theorem population is completed-active production within the fixed window. Raw history and attempted-only records are reported but are not claimed as the theorem arrival stream. The live theorem dispatch mode should run with SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block when future tasks are to be admitted into the theorem population.

## Strict-Classifier Non-Matches

These rows are service-domain admitted but do not have a strict
production classifier rule. They should be described as admission
certified, not as strict classifier coverage.

| Task | Status | Project | Classifier reason | Inferred workload |
|---|---|---|---|---|
| `t10018` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10038` | `done` | `Asumption Agent` | `unmapped_cpu` | `assumption_agent_live_benchmark_completed_history` |
| `t10124` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10233` | `done` | `scheduleurm` | `unmapped_cpu` | `scheduleurm_control_plane_completed_history` |
| `t10235` | `queued` | `TransitDuet` | `unmapped_cpu` | `transit_freqhrl_import_smoke_c_le2_completed_history` |
