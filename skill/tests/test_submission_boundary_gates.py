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
