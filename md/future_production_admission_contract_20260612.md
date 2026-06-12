# Future Production Admission Contract

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `active_count` | 36 |
| `active_production_count` | 36 |
| `admitted_traceable_count` | 36 |
| `probe_required_count` | 0 |
| `future_production_automatic_theorem_closure_ready` | true |
| `future_jobs_all_theorem_grade_without_probe` | false |

## Route Counts

| Route | Count |
|---|---:|
| `ADMIT_THEOREM_TRACE` | 36 |

## Active Production Sample

| Task | Status | Project | Workload | Route | Reason |
|---|---|---|---|---|---|
| `t10263` | `running` | `scheduleurm` | `scheduleurm_control_plane_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10370` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10371` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10372` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10373` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10374` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10375` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10376` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10377` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10378` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10379` | `running` | `BAMOR` | `bamor_mujoco_c3_8_completed_history` | `ADMIT_THEOREM_TRACE` | `` |
| `t10388` | `running` | `FreqDuet` | `freqduet_cpu_ablation_c9_16` | `ADMIT_THEOREM_TRACE` | `` |
| `t10391` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10392` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10393` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10394` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10395` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10396` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10397` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10398` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10399` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10400` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10401` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10402` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10403` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10404` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10405` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10406` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10407` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10408` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10409` | `queued` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10410` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10411` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10412` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10413` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |
| `t10414` | `running` | `BAPR` | `hybrid_rl_resac_ant` | `ADMIT_THEOREM_TRACE` | `` |

## Scope

Future production closure is automatic only as an admission contract: new jobs are classified, admitted if measured, and otherwise routed to probe.  The certificate prevents unmeasured future jobs from silently entering theorem-facing claims.
