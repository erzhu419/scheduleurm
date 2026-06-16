# Declared Finite-Domain Positive-Cover Gate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `DECLARED_FINITE_POSITIVE_COVER_PASS_ALL_STATE_POSITIVE_FALSE` |
| `gate_pass` | true |
| `scoped_claim_ready` | true |
| `strong_claim_ready` | false |
| `pass_meaning` | classification and positive-cover certificate over the declared finite service-cache domain, not arbitrary all-state positive service |
| `declared_finite_positive_cover_ready` | true |
| `positive_service_all_state_cover_ready` | false |
| `declared_universe_bucket_count` | 225 |
| `workload_domain_count` | 120 |
| `positive_bucket_count` | 215 |
| `boundary_bucket_count` | 10 |
| `probe_required_bucket_count` | 0 |
| `uncovered_bucket_count` | 0 |
| `declared_domain_classification_fraction` | 1.000000 |
| `positive_service_fraction` | 0.955556 |
| `boundary_fraction` | 0.044444 |
| `uncovered_fraction` | 0.000000 |

This gate passes the scoped certificate. It does not make the adjacent strong claim.

## Status Counts

| Status | Count |
|---|---:|
| `capacity_boundary` | 10 |
| `positive_lower_service` | 215 |

## Declared Universe

service_cache workload_key x exact measured profile x node_bucket x resource_kind x command_fingerprint

## Scope

Closes a declared finite-domain positive cover for exact measured service-cache buckets.  It does not certify arbitrary future states; unknown states must enter probe/admission before joining the positive theorem population.
