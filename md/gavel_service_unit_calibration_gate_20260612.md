# Gavel Service-Unit Calibration Gate

This artifact is reproducible, but the calibration claim is ready only when `gate_pass=true` and `scoped_claim_ready=true`. Current pending rows must not be read as service-unit equivalence.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `GAVEL_PROFILE_AWARE_MODEL_CALIBRATION_PASS_SCALAR_PENDING` |
| `gate_pass` | true |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `gavel_service_unit_equivalence_ready` | false |
| `profile_aware_model_calibration_ready` | true |
| `direct_full_stack_same_workload_ready` | true |
| `relative_error_threshold` | 0.05 |
| `min_holdout_rows_per_taskset` | 3 |
| `paired_source_row_count` | 12 |
| `paired_holdout_path` | `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/gavel_service_unit_paired_holdout_20260612.json` |

## Rows

| Taskset | Native jobs | Native avg JCT | Native makespan | Scheduleurm best profile | Scheduleurm makespan | Implied makespan scale | Calibration rows | Holdout rows | Scalar p95 | Profile-aware p95 | Scalar ready | Profile ready |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q01_gpu_bound_compute` | 4 | 25.464 | 25.464 | 1 | 102.541 | 4.02689 | 3 | 3 | 0.5 | 0 | false | true |
| `q11_cpu_gpu_coupled` | 8 | 11.149 | 11.149 | 2 | 1920 | 172.213 | 3 | 3 | 0.500022 | 0 | false | true |

## Blocker

none for the scoped profile-aware same-workload native Gavel simulator baseline; scalar service-unit equivalence remains false because the single-scale q01/q11 p95 relative errors exceed the threshold

## Next Threshold

To promote the adjacent scalar-equivalence claim, replace the profile-aware service-curve model by a single service-unit scale whose q01/q11 holdout p95 relative error is <= 0.05 with at least 3 holdout rows per taskset.

## Scope

Audit gate for Gavel calibration.  The scalar service-unit equivalence claim is ready only when gavel_service_unit_equivalence_ready=true.  The profile-aware model claim is a narrower same-workload native Gavel simulator baseline on Scheduleurm measured service curves.
