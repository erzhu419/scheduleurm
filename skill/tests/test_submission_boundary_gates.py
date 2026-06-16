from algorithm.experiments.future_admitted_fabric_cover_gate import (
    build_future_admitted_fabric_cover_gate,
)
from algorithm.experiments.future_production_admission_contract import (
    build_future_production_admission_contract,
)
from algorithm.experiments.q00_q10_broad_envelope_gate import (
    build_q00_q10_broad_envelope_gate,
)
from algorithm.experiments.all_state_conservative_cover_gate import (
    build_all_state_conservative_cover_gate,
)
from algorithm.experiments.sota_fullstack_superiority_gate import (
    build_sota_fullstack_superiority_gate,
)
from algorithm.experiments.sota_candidate_union_gate import (
    build_sota_candidate_union_gate,
)
from algorithm.experiments.sota_native_execution_attempts import (
    build_sota_native_execution_attempts,
)
from algorithm.experiments.sota_algorithm_upgrade_gate import (
    build_sota_algorithm_upgrade_gate,
)
from algorithm.experiments.gavel_service_unit_equivalence_certificate import (
    build_gavel_service_unit_equivalence_certificate,
)
from algorithm.experiments.gavel_service_unit_calibration_gate import (
    build_gavel_service_unit_calibration_gate,
)
from algorithm.experiments.declared_finite_domain_positive_cover_gate import (
    build_declared_finite_domain_positive_cover_gate,
)
from algorithm.experiments.controlled_production_completion_gate import (
    build_controlled_production_completion_gate,
)
from algorithm.experiments.organic_production_canary_recorder_gate import (
    build_organic_production_canary_recorder_gate,
)
from algorithm.experiments.organic_history_completion_gate import (
    build_organic_history_completion_gate,
)
from algorithm.experiments.global_theorem_dispatcher_prototype_gate import (
    build_global_theorem_dispatcher_prototype_gate,
)
from algorithm.experiments.reviewer_environment_manifest_gate import (
    build_reviewer_environment_manifest_gate,
)
from algorithm.experiments.online_ablation_summary_ci import (
    build_online_ablation_summary_ci,
)
from algorithm.experiments.selected_profile_holdout_lcb_gate import (
    build_selected_profile_holdout_lcb_gate,
)
from algorithm.experiments.corner_case_lower_service_gate import (
    build_corner_case_lower_service_gate,
)
from algorithm.experiments.gavel_resident_delay_jct_holdout_gate import (
    build_gavel_resident_delay_jct_holdout_gate,
)
from algorithm.experiments.sota_universe_registry_gate import (
    build_sota_universe_registry_gate,
)
from algorithm.experiments.sota_admitted_universe_closure_gate import (
    build_sota_admitted_universe_closure_gate,
)
from algorithm.experiments.registered_sota_runtime_gate import (
    build_registered_sota_runtime_gate,
)
from algorithm.experiments.registered_sota_adapter_closure_gate import (
    build_registered_sota_adapter_closure_gate,
)
from algorithm.experiments.future_workload_protocol_gate import (
    build_future_workload_protocol_gate,
)
from algorithm.experiments.multinode_original_deployment_gate import (
    build_multinode_original_deployment_gate,
)
from algorithm.experiments.multinode_theorem_shadow_gate import (
    build_multinode_theorem_shadow_gate,
)
from algorithm.experiments.multinode_history_completion_gate import (
    build_multinode_history_completion_gate,
)
from algorithm.experiments.decima_spark_dag_gate import (
    build_decima_spark_dag_gate,
)
from algorithm.experiments.decima_same_domain_benchmark_gate import (
    build_decima_same_domain_benchmark_gate,
)
from algorithm.experiments.production_wide_organic_trace_gate import (
    build_production_wide_organic_trace_gate,
)
from algorithm.experiments.production_organic_readiness_bridge_gate import (
    build_production_organic_readiness_bridge_gate,
)
from algorithm.experiments.universal_claim_closure_gate import (
    build_universal_claim_closure_gate,
)
from algorithm.experiments.non_future_claim_closure_gate import (
    build_non_future_claim_closure_gate,
)
from algorithm.theorem_dispatch.service_registry import infer_workload_key


def test_future_admitted_fabric_cover_is_measured_state_not_all_state():
    report = build_future_admitted_fabric_cover_gate()

    assert report["future_admitted_measured_state_ready"] is True
    assert report["future_all_state_fabric_cover_ready"] is False
    assert report["pass"] is True


def test_future_production_admission_routes_unmeasured_jobs_to_probe():
    rows = [
        {
            "id": "known",
            "status": "queued",
            "project": "BAPR",
            "signature": "BAPR/resac/train",
            "cmd": "python train.py --env Ant-v5",
            "est_vram_mb": 12000,
        },
        {
            "id": "unknown",
            "status": "queued",
            "project": "new_project",
            "signature": "totally/new/workload",
            "cmd": "python future_unmeasured_program.py",
            "est_vram_mb": 0,
        },
    ]

    report = build_future_production_admission_contract(records=rows)
    routes = {row["task_id"]: row["route"] for row in report["rows"]}
    rows_by_id = {row["task_id"]: row for row in report["rows"]}

    assert routes["known"] == "ADMIT_THEOREM_TRACE"
    assert routes["unknown"] == "PROBE_REQUIRED"
    assert rows_by_id["known"]["theorem_admission_route"] == "ADMIT_THEOREM_TRACE"
    assert rows_by_id["unknown"]["theorem_admission_route"] == "PROBE_REQUIRED"
    assert rows_by_id["unknown"]["theorem_workload_key"] == ""
    assert report["future_production_automatic_theorem_closure_ready"] is True
    assert report["future_jobs_all_theorem_grade_without_probe"] is False
    assert report["theorem_admission_default_trace_mode_ready"] is True


def test_q00_q10_broad_gate_is_measured_envelope_not_all_world():
    report = build_q00_q10_broad_envelope_gate()

    assert report["broad_measured_q00_q10_envelope_ready"] is True
    assert report["broad_all_cpu_dataloader_world_ready"] is False
    assert report["q00_workload_count"] >= 1
    assert report["q10_workload_count"] >= 1


def test_all_state_conservative_cover_keeps_positive_unknown_load_out_of_claim():
    report = build_all_state_conservative_cover_gate()

    assert report["all_state_safety_cover_ready"] is True
    assert report["positive_service_all_state_cover_ready"] is False
    assert report["all_state_stability_for_unknown_positive_arrivals_ready"] is False


def test_sota_fullstack_gate_is_named_runtime_probe_not_sota_superiority():
    report = build_sota_fullstack_superiority_gate()

    assert report["hard_blocker_certificate_ready"] is True
    assert report["named_same_host_runtime_probe_ready"] is True
    assert report["named_same_host_runtime_probe_native_better"] is True
    assert report["direct_fullstack_sota_superiority_ready"] is False
    assert report["direct_fullstack_named_sota_superiority_ready"] is False
    assert report["registered_sota_universe_superiority_ready"] is False
    assert report["arbitrary_sota_superiority_ready"] is False
    assert report["multinode_original_deployment_superiority_ready"] is False
    assert report["production_wide_organic_trace_superiority_ready"] is False
    assert report["named_same_host_runtime_probe_ready_count"] == 5
    assert report["named_same_host_runtime_probe_native_better_count"] == 5
    assert report["policy_semantics_comparison_ready"] is True
    assert report["gavel_resident_delay_jct_holdout"]["ready"] is True
    assert "direct full-stack SOTA superiority" in report["scope"]


def test_sota_candidate_union_beats_policy_envelopes_without_fullstack_claim():
    report = build_sota_candidate_union_gate(
        tasksets=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled"),
        arrivals=("static",),
    )
    aggregate = report["aggregate"]

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert aggregate["union_makespan_beats_sota_envelope_all"] is True
    assert aggregate["union_mean_flow_beats_sota_envelope_all"] is True
    assert aggregate["guarded_union_not_pareto_dominated_all"] is True
    assert aggregate["pareto_slack_union_beats_both_envelopes_all"] is True
    assert aggregate["fixed_online_policy_pareto_dominates_sota_style_all"] is True
    assert any(
        row["policy_family"] == "candidate_union"
        for row in aggregate["policy_aggregate"]
    )


def test_sota_native_execution_attempts_keep_direct_fullstack_blocker_visible():
    report = build_sota_native_execution_attempts(run_commands=False)

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["direct_fullstack_sota_superiority_ready"] is False
    assert report["direct_full_stack_ready_count"] == 0


def test_sota_algorithm_upgrade_gate_closes_five_scoped_axes():
    report = build_sota_algorithm_upgrade_gate()

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["adaptive_scalarized_single_policy"]["adaptive_scalarized_ready"] is True
    assert report["adaptive_scalarized_single_policy"]["not_pareto_dominated_all"] is True
    assert report["pareto_slack_fixed_online_policy"]["pareto_slack_ready"] is True
    assert report["pareto_slack_fixed_online_policy"]["fixed_online_policy_pareto_dominates_sota_style_all"] is True
    assert report["state_dependent_marginal_cache"]["state_dependent_marginal_cache_ready"] is True
    assert report["global_batch_lookahead"]["lookahead_global_batch_ready"] is True
    assert report["eta_reuse"]["eta_reuse_ready"] is True
    assert report["expanded_sota_policy_families"]["expanded_sota_families_ready"] is True


def test_gavel_service_unit_certificate_keeps_unit_blocker_visible():
    report = build_gavel_service_unit_equivalence_certificate()

    assert report["trace_schema_compatibility_ready"] is True
    assert report["throughput_seed_ready"] is True
    assert report["gavel_native_bounded_microbaseline_ready"] is True
    assert report["gavel_service_unit_equivalence_ready"] is False
    assert report["direct_full_stack_same_workload_ready"] is False
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert "SERVICE_UNIT_EQUIV_FALSE" in report["status"]


def test_gavel_service_unit_calibration_gate_passes_profile_model_only():
    report = build_gavel_service_unit_calibration_gate()

    assert report["pass"] is True
    assert report["gate_pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["profile_aware_model_calibration_ready"] is True
    assert report["gavel_service_unit_equivalence_ready"] is False
    assert report["direct_full_stack_same_workload_ready"] is True
    assert report["status"] == "GAVEL_PROFILE_AWARE_MODEL_CALIBRATION_PASS_SCALAR_PENDING"
    assert report["paired_source_row_count"] >= 12
    assert report["rows"]


def test_gavel_resident_delay_jct_holdout_is_scoped_not_fullstack():
    report = build_gavel_resident_delay_jct_holdout_gate()

    assert report["gavel_style_resident_delay_jct_holdout_ready"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["usable_row_count"] == 3
    assert report["admit_now_jct_better_count"] == 3
    assert report["admit_now_makespan_better_count"] == 3
    assert all(row["resident_alone_challenge_rate_step_s"] > 0.0 for row in report["rows"])


def test_corner_case_lower_service_gate_admits_standalone_cache_only():
    report = build_corner_case_lower_service_gate()

    assert report["corner_case_service_cache_ready"] is True
    assert report["corner_case_lower_service_capacity_ready"] is True
    assert report["co_location_lower_service_capacity_ready"] is True
    assert report["cache_roundtrip_ready"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["service_cache_record_count"] == 5


def test_declared_finite_domain_cover_is_not_arbitrary_all_state():
    report = build_declared_finite_domain_positive_cover_gate()

    assert report["declared_finite_positive_cover_ready"] is True
    assert report["positive_service_all_state_cover_ready"] is False
    assert report["uncovered_bucket_count"] == 0
    assert report["positive_bucket_count"] > 0
    assert report["declared_domain_classification_fraction"] == 1.0
    assert 0.0 < report["positive_service_fraction"] < 1.0
    assert report["strong_claim_ready"] is False


def test_strict_theorem_admission_disables_token_overlap_fallback():
    fuzzy_task = {
        "project": "future",
        "signature": "freqduet cpu ablation unknown",
        "cmd": "python totally_new.py --freqduet --cpu --ablation",
    }
    explicit_task = {
        "project": "future",
        "workload_key": "gpu_heavy_jax_matmul",
        "cmd": "python totally_new.py",
    }

    assert infer_workload_key(fuzzy_task)
    assert infer_workload_key(fuzzy_task, admission_mode="strict") == ""
    assert infer_workload_key(explicit_task, admission_mode="strict") == "gpu_heavy_jax_matmul"


def test_controlled_completion_gate_splits_controlled_history_and_live_trace_claims():
    report = build_controlled_production_completion_gate(allow_launch=False)

    assert report["bounded_controlled_completion_ready"] is True
    assert report["organic_production_canary_recorder_ready"] is True
    assert report["controlled_6_task_completion_ready"] is True
    assert report["controlled_32_task_completion_ready"] is True
    assert report["large_scale_organic_launched_completion_ready"] is True
    assert report["strict_history_large_scale_completion_ready"] is True
    assert report["live_oracle_traced_large_scale_completion_ready"] is False
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is True
    assert report["status"] == "CONTROLLED_32_AND_STRICT_HISTORY_COMPLETION_PASS"


def test_organic_production_canary_recorder_is_not_completion_claim(tmp_path):
    trace_path = tmp_path / "empty_oracle_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    report = build_organic_production_canary_recorder_gate(
        trace_path=trace_path,
        use_history_completion=False,
    )

    assert report["organic_production_canary_recorder_ready"] is True
    assert report["large_scale_organic_launched_completion_ready"] is False
    assert report["strong_claim_ready"] is False
    assert "COMPLETION_PENDING" in report["status"]


def test_organic_history_completion_gate_closes_strict_history_population():
    report = build_organic_history_completion_gate()

    assert report["large_scale_organic_history_completion_ready"] is True
    assert report["large_scale_organic_launched_completion_ready"] is True
    assert report["strict_organic_launched_count"] >= 64
    assert report["strict_organic_completed_count"] >= 50
    assert report["strict_organic_completion_fraction"] >= 0.80
    assert report["strict_unadmitted_launched_count"] == 0
    assert report["workload_domain_count"] >= 3
    assert report["node_count"] >= 2


def test_online_ablation_summary_reports_distributional_counts():
    report = build_online_ablation_summary_ci()

    assert report["online_scenario_count"] == 192
    assert report["ablation_scenario_count"] > 0
    assert report["online_summary"]["candidate_vs_legacy_makespan"]["n"] == 192
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False


def test_selected_profile_holdout_lcb_gate_passes_lower_service_only():
    report = build_selected_profile_holdout_lcb_gate()

    assert report["pass"] is True
    assert report["gate_pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["selected_profile_stochastic_lcb_ready"] is True
    assert report["usable_for_stochastic_holdout_theorem"] is True
    assert report["lcb_lower_service_capacity_ready"] is True
    assert report["absolute_mean_load_eta_ready"] is False
    assert report["diagonal_normalized_eta_ready"] is False
    assert report["strong_claim_ready"] is False
    assert report["target_count"] >= 11
    assert report["sample_pending_count"] == 0
    assert report["epsilon_gap_to_positive_eta"] > 0.0


def test_global_theorem_dispatcher_prototype_excludes_legacy_rows():
    report = build_global_theorem_dispatcher_prototype_gate()
    result = report["result"]

    assert report["global_action_dispatch_ready"] is True
    assert report["live_scheduler_default_global_dispatcher_ready"] is False
    assert result["oracle_gap_alpha0"] == 0.0
    assert result["oracle_gap_alpha1"] == 0.0
    assert result["fallback_legacy_rows_excluded"] is True


def test_reviewer_environment_manifest_is_not_container_claim():
    report = build_reviewer_environment_manifest_gate()

    assert report["scoped_claim_ready"] is True
    assert report["clean_container_ready"] is False
    assert report["strong_claim_ready"] is False
    assert report["make_targets_ready"] is True


def test_sota_universe_gate_closes_named_not_arbitrary_sota():
    report = build_sota_universe_registry_gate()

    assert report["named_five_runtime_probe_ready"] is True
    assert report["named_five_fullstack_superiority_ready"] is False
    assert report["scoped_claim_ready"] is True
    assert report["registered_extension_runtime_inventory_ready"] is True
    assert report["registered_sota_universe_superiority_ready"] is False
    assert report["arbitrary_sota_superiority_ready"] is False
    assert report["strong_claim_ready"] is False


def test_sota_admitted_universe_closes_policy_semantics_not_arbitrary_sota():
    report = build_sota_admitted_universe_closure_gate()

    assert report["scoped_claim_ready"] is True
    assert report["registered_sota_policy_semantics_universe_ready"] is True
    assert report["registered_sota_admitted_policy_superiority_ready"] is True
    assert report["registered_sota_universe_superiority_ready"] is False
    assert report["arbitrary_sota_superiority_ready"] is False
    assert report["strong_claim_ready"] is False
    assert report["policy_semantics_admitted_count"] == report["non_adjacent_registered_system_count"]


def test_registered_sota_runtime_gate_is_inventory_not_fullstack():
    report = build_registered_sota_runtime_gate()

    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["registered_extension_system_count"] == 6
    assert report["runnable_public_runtime_system_count"] == 4
    assert report["same_workload_fullstack_ready_count"] == 0
    rows = {row["system"]: row for row in report["rows"]}
    assert rows["Tiresias"]["repo_or_paper_inventory_ready"] is True
    assert rows["Shockwave"]["repo_or_paper_inventory_ready"] is True
    assert rows["AlloX"]["repo_or_paper_inventory_ready"] is True
    assert rows["Optimus"]["repo_or_paper_inventory_ready"] is True
    assert rows["Themis"]["has_public_runtime_repo"] is False
    assert rows["Gandiva"]["has_public_runtime_repo"] is False


def test_registered_sota_adapter_gate_closes_policy_universe_not_direct_binary():
    report = build_registered_sota_adapter_closure_gate()

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["registered_system_count"] == 11
    assert report["policy_service_unit_adapter_ready_count"] == report["registered_system_count"]
    assert report["registered_systems_without_any_policy_adapter_count"] == 0
    assert report["registered_policy_adapter_universe_ready"] is True
    assert report["registered_direct_external_binary_superiority_ready"] is False
    assert report["direct_external_binary_adapter_ready_count"] < report["registered_system_count"]
    assert "Tiresias" in report["registered_systems_without_direct_binary_adapter"]


def test_future_workload_protocol_routes_unknown_future_jobs_to_probe():
    report = build_future_workload_protocol_gate()

    assert report["future_workload_protocol_ready"] is True
    assert report["unknown_future_jobs_probe_required_ready"] is True
    assert report["future_admitted_measured_state_ready"] is True
    assert report["arbitrary_future_workload_theorem_ready"] is False
    assert report["strong_claim_ready"] is False
    assert report["synthetic_known_count"] >= 5
    assert report["synthetic_unknown_count"] >= 5
    assert report["synthetic_routes"]["future_unknown_llm_new_arch"] == "PROBE_REQUIRED"
    assert report["synthetic_routes"]["future_unknown_cpu_pipeline"] == "PROBE_REQUIRED"
    assert report["synthetic_routes"]["future_unknown_spark_dag"] == "PROBE_REQUIRED"


def test_multinode_original_deployment_gate_is_not_same_host_claim():
    report = build_multinode_original_deployment_gate(probe_remotes=False)

    assert report["same_host_named_runtime_probe_ready"] is True
    assert report["same_host_named_fullstack_ready"] is False
    assert report["scoped_claim_ready"] is True
    assert report["scheduleurm_multinode_history_completion_ready"] is True
    assert report["original_multinode_deployment_rows_ready"] is True
    assert report["multinode_original_deployment_superiority_ready"] is False
    assert report["strong_claim_ready"] is False


def test_multinode_history_completion_gate_closes_scheduleurm_native_history():
    report = build_multinode_history_completion_gate()

    assert report["scheduleurm_multinode_launched_completion_ready"] is True
    assert report["strict_launched_count"] >= 64
    assert report["strict_completed_count"] >= 50
    assert report["strict_unadmitted_count"] == 0
    assert report["node_count"] >= 2
    assert report["gpu_node_count"] >= 2
    assert report["workload_domain_count"] >= 3


def test_multinode_theorem_shadow_without_probe_is_not_launch_claim():
    report = build_multinode_theorem_shadow_gate(
        probe_remotes=False,
        nodes=("jtl110gpu", "jtl110gpu2"),
    )

    assert report["strong_claim_ready"] is False
    assert report["multinode_original_launch_claim_ready"] is False
    assert report["probe_remotes"] is False


def test_decima_gate_separates_spark_dag_from_gpu_colocation():
    report = build_decima_spark_dag_gate()

    assert report["repo_ready"] is True
    assert report["spark_dag_simulator_smoke_ready"] is True
    assert report["spark_dag_heuristic_benchmark_ready"] is True
    assert report["spark_dag_metric_bridge_ready"] is True
    assert report["scoped_claim_ready"] is True
    assert report["scheduleurm_gpu_colocation_comparable_ready"] is False
    assert report["direct_fullstack_gpu_sota_claim_ready"] is False
    assert report["strong_claim_ready"] is False


def test_decima_same_domain_benchmark_is_execution_not_gpu_superiority():
    report = build_decima_same_domain_benchmark_gate()

    assert report["pass"] is True
    assert report["spark_dag_same_domain_benchmark_ready"] is True
    assert report["spark_dag_same_domain_execution_ready"] is True
    assert report["completed_pair_count"] == report["seed_count"]
    assert report["scheduleurm_gpu_colocation_comparable_ready"] is False
    assert report["direct_fullstack_gpu_sota_claim_ready"] is False
    assert report["strong_claim_ready"] is False


def test_production_wide_organic_trace_gate_closes_history_completion_path():
    report = build_production_wide_organic_trace_gate()

    assert report["scoped_claim_ready"] is True
    assert report["organic_production_canary_recorder_ready"] is True
    assert report["production_wide_live_trace_closed"] is False
    assert report["production_wide_history_completion_closed"] is True
    assert report["large_scale_organic_launched_completion_ready"] is True
    assert report["strong_claim_ready"] is True


def test_production_organic_readiness_bridge_uses_history_completion_snapshot():
    report = build_production_organic_readiness_bridge_gate(min_shadow_tasks=0)

    assert report["scoped_claim_ready"] is True
    assert report["organic_readiness_bridge_ready"] is True
    assert report["history_large_scale_organic_launched_completion_ready"] is True
    assert report["large_scale_organic_launched_completion_ready"] is True
    assert report["strong_claim_ready"] is True


def test_universal_claim_closure_gate_is_boundary_not_universal_theorem():
    report = build_universal_claim_closure_gate(probe_remotes=False)

    assert report["pass"] is True
    assert report["scoped_claim_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["scoped_ready_count"] == 5
    assert report["strong_ready_count"] == 1
    assert {row["name"] for row in report["rows"]} == {
        "arbitrary_sota_universe",
        "arbitrary_future_workload",
        "multinode_original_deployment",
        "decima_spark_dag",
        "production_wide_organic_trace",
    }


def test_non_future_claim_closure_gate_closes_finite_nonfuture_scope():
    report = build_non_future_claim_closure_gate(probe_remotes=False)

    assert report["pass"] is True
    assert report["non_future_scoped_closure_ready"] is True
    assert report["strong_claim_ready"] is False
    assert report["excluded_direction"] == "arbitrary_future_workload"
    assert report["ready_count"] == report["row_count"]
    rows = {row["name"]: row for row in report["rows"]}
    assert rows["registered_sota_policy_semantics_universe"]["ready"] is True
    assert rows["production_wide_organic_completion"]["ready"] is True
    assert rows["scheduleurm_multinode_fullstack_history"]["ready"] is True
    assert rows["decima_spark_dag_bridge"]["ready"] is True
