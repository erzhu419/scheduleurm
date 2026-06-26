# Organic History Completion Gate

This gate audits real scheduler history under strict exact/signature service admission. It excludes raw-history rows that are not controlled theorem-facing production rows.

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `status` | `ORGANIC_HISTORY_COMPLETION_PASS` |
| `strong_claim_ready` | true |
| `history_large_scale_organic_completion_ready` | true |
| `live_trace_large_scale_organic_completion_ready` | false |
| `combined_large_scale_completion_evidence_ready` | true |
| `raw_history_closed` | false |
| `admission_mode` | `strict_exact_or_signature_service_certificate` |
| `raw_production_row_count` | 10134 |
| `raw_production_launched_count` | 5960 |
| `raw_production_completed_count` | 5949 |
| `excluded_row_count` | 2649 |
| `excluded_launched_count` | 1902 |
| `strict_organic_launched_count` | 4058 |
| `strict_organic_completed_count` | 4047 |
| `strict_organic_completion_fraction` | 0.9972893050763924 |
| `strict_unadmitted_launched_count` | 0 |
| `workload_domain_count` | 35 |
| `node_count` | 13 |
| `project_count` | 21 |

## Top Workload Domains

| Workload key | Count |
|---|---:|
| `gpu_heavy_jax_matmul` | 1166 |
| `bamor_mujoco_c3_8_completed_history` | 679 |
| `hybrid_rl_resac_ant` | 404 |
| `h2oplus_shell_eval_c_le2_completed_history` | 398 |
| `freqduet_cpu_ablation_c9_16` | 254 |
| `transit_native_promotion_c3_8_persistent_stress_completed_history` | 249 |
| `freqduet_cpu_ablation_c17_32` | 234 |
| `transit_freqhrl_import_smoke_c_le2_completed_history` | 136 |
| `freqduet_cpu_ablation_c33_64_completed_history` | 75 |
| `offline_sumo_eval_c33_64_completed_history` | 65 |
| `bamor_train_compare_c3_8_completed_history` | 58 |
| `sumo_eval_simple_sac_c_le2` | 53 |
| `freqduet_cpu_ablation_c3_8_completed_history` | 41 |
| `h2oplus_snapshot_gpu_completed_history` | 38 |
| `bamor_diagnostic_shard_c9_16_completed_history` | 34 |
| `transit_native_promotion_c9_16_wait_credit_shell_completed_history` | 28 |
| `bamor_diagnostic_shard_c3_8_completed_history` | 25 |
| `freqduet_snapshot_counterfactual_c9_16_completed_history` | 24 |
| `bamor_diagnostic_shard_c17_32_completed_history` | 17 |
| `transit_freqhrl_analysis_matrix_c_le2_completed_history` | 16 |

## Exclusion Reasons

| Reason | Count |
|---|---:|
| `auto_adopted` | 2212 |
| `nonproduction_project:md` | 1 |
| `nonproduction_project:proof` | 2 |
| `nonproduction_project:python` | 2 |
| `nonproduction_project:sched` | 2 |
| `nonproduction_project:tmp` | 424 |
| `outside_scheduler_adopted` | 6 |

## Blocker

none

## Scope

Real scheduler history audit for the strict theorem-facing organic production population.  Raw history, attempted-only jobs, cancelled jobs, auto-adopted jobs, diagnostic probes, and read-only probes are reported but excluded.  The claim is not about arbitrary future workloads; future unknown jobs still require the admission/probe contract.  This is a strict scheduler-history certificate, not a large live-trace completion certificate.
