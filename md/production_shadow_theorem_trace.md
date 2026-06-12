# Production Shadow Theorem Trace

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `production_shadow_trace_closed` | true |
| `launched_dispatch_claim` | false |
| `active_production_shadow_task_count` | 24 |
| `placed_count` | 14 |
| `unplaced_count` | 10 |
| `trace_slot_count` | 14 |
| `theorem_slot_count` | 12 |
| `candidate_count_total` | 58 |
| `bridge_usable_for_theorem` | false |
| `theorem_subset_bridge_usable` | true |
| `alpha0` | NA |
| `alpha1` | NA |
| `theorem_subset_alpha0` | 0.000000000 |
| `theorem_subset_alpha1` | 0.000000000 |

## Scope

non-invasive shadow trace over current active production tasks. It reads queue/node state and evaluates the optional theorem hook without queue mutation or launch.  It is not a production-wide launched dispatch trace.

## Non-Theorem Blockers

| Reason | Count |
|---|---:|
| `scheduler_sort_key_minimization` | 2 |

## Placements

| Shadow task | Source task | Project | Workload | Admitted | Placement |
|---|---|---|---|---:|---|
| `shadow-production-0000-t9748` | `t9748` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 1, "node": "jtl110gpu2"}` |
| `shadow-production-0001-t9749` | `t9749` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "jtl110gpu2"}` |
| `shadow-production-0002-t9755` | `t9755` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "jtl110gpu2"}` |
| `shadow-production-0003-t9915` | `t9915` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "jtl110gpu2"}` |
| `shadow-production-0004-t9917` | `t9917` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `shadow-production-0005-t9919` | `t9919` | `CS-BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "node007-direct"}` |
| `shadow-production-0006-t10086` | `t10086` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 1, "node": "node007-direct"}` |
| `shadow-production-0007-t10087` | `t10087` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 1, "node": "node007-direct"}` |
| `shadow-production-0008-t10088` | `t10088` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 2, "node": "node007-direct"}` |
| `shadow-production-0009-t10089` | `t10089` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 2, "node": "node007-direct"}` |
| `shadow-production-0010-t10090` | `t10090` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 0, "node": "jtl110gpu"}` |
| `shadow-production-0011-t10091` | `t10091` | `BAPR` | `hybrid_rl_resac_ant` | true | `{"gpu_idx": 1, "node": "jtl110gpu"}` |
| `shadow-production-0012-t10263` | `t10263` | `scheduleurm` | `scheduleurm_control_plane_completed_history` | true | `{"gpu_idx": null, "node": "local"}` |
| `shadow-production-0013-t10335` | `t10335` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0014-t10336` | `t10336` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0015-t10337` | `t10337` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0016-t10338` | `t10338` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0017-t10339` | `t10339` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0018-t10340` | `t10340` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0019-t10341` | `t10341` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0020-t10342` | `t10342` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0021-t10343` | `t10343` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0022-t10344` | `t10344` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | true | `null` |
| `shadow-production-0023-t10348` | `t10348` | `Asumption Agent` | `assumption_agent_unittest_completed_history` | true | `{"gpu_idx": null, "node": "local"}` |
