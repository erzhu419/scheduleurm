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
from algorithm.experiments.gavel_service_unit_equivalence_certificate import (
    build_gavel_service_unit_equivalence_certificate,
)
from algorithm.experiments.declared_finite_domain_positive_cover_gate import (
    build_declared_finite_domain_positive_cover_gate,
)
from algorithm.experiments.controlled_production_completion_gate import (
    build_controlled_production_completion_gate,
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

    assert routes["known"] == "ADMIT_THEOREM_TRACE"
    assert routes["unknown"] == "PROBE_REQUIRED"
    assert report["future_production_automatic_theorem_closure_ready"] is True
    assert report["future_jobs_all_theorem_grade_without_probe"] is False


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


def test_sota_fullstack_superiority_gate_does_not_promote_scaffold_to_claim():
    report = build_sota_fullstack_superiority_gate()

    assert report["hard_blocker_certificate_ready"] is True
    assert report["direct_fullstack_sota_superiority_ready"] is False
    assert report["policy_semantics_comparison_ready"] is True


def test_gavel_service_unit_certificate_keeps_unit_blocker_visible():
    report = build_gavel_service_unit_equivalence_certificate()

    assert report["trace_schema_compatibility_ready"] is True
    assert report["throughput_seed_ready"] is True
    assert report["gavel_native_bounded_microbaseline_ready"] is True
    assert report["gavel_service_unit_equivalence_ready"] is False
    assert report["direct_full_stack_same_workload_ready"] is False


def test_declared_finite_domain_cover_is_not_arbitrary_all_state():
    report = build_declared_finite_domain_positive_cover_gate()

    assert report["declared_finite_positive_cover_ready"] is True
    assert report["positive_service_all_state_cover_ready"] is False
    assert report["uncovered_bucket_count"] == 0
    assert report["positive_bucket_count"] > 0


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


def test_controlled_completion_gate_separates_bounded_from_32_task_claim():
    report = build_controlled_production_completion_gate(allow_launch=False)

    assert report["bounded_controlled_completion_ready"] is True
    assert report["organic_production_canary_recorder_ready"] is True
    assert report["controlled_32_task_completion_ready"] is False
    assert report["large_scale_organic_launched_completion_ready"] is False
