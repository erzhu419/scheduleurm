# Future Production Admission Contract

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `STRICT_TRACE_ADMISSION_CONTRACT_PASS_UNKNOWN_PROBE_REQUIRED` |
| `admission_mode` | `strict` |
| `theorem_admission_default_trace_mode_ready` | true |
| `active_count` | 11 |
| `active_production_count` | 11 |
| `admitted_traceable_count` | 11 |
| `probe_required_count` | 0 |
| `future_production_automatic_theorem_closure_ready` | true |
| `future_jobs_all_theorem_grade_without_probe` | false |

## Route Counts

| Route | Count |
|---|---:|
| `ADMIT_THEOREM_TRACE` | 11 |

## Active Production Sample

| Task | Status | Project | Workload | Route | Reason | Source |
|---|---|---|---|---|---|---|
| `t11571` | `running` | `BAPR-BUS` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11572` | `running` | `CS-BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11573` | `running` | `CS-BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11574` | `running` | `CS-BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11575` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11576` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11577` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11578` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11579` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11580` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |
| `t11581` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `exact_positive_service_domain_certified` | `service_cache` |

## Scope

Future production closure is automatic only as an admission contract: new jobs are classified, admitted if measured, and otherwise routed to probe.  The certificate prevents unmeasured future jobs from silently entering theorem-facing claims.
