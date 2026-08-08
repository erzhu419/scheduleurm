from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_benchmark import (
    OURS_POLICY,
    POLICY_ORDER,
    build_mmrcpsp_benchmark,
    schedule_instance,
    write_artifacts,
)
from algorithm.experiments.mmrcpsp_instances import (
    MMRCPSPFormatError,
    feasible_mode_actions,
    parse_psplib_mm,
    parse_psplib_mm_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "tests" / "data"
CORE_INSTANCE = DATA_ROOT / "mmrcpsp_core.mm"
RESERVE_INSTANCE = DATA_ROOT / "mmrcpsp_nonrenewable_reserve.mm"


def test_parser_reads_psplib_precedence_modes_and_resource_kinds():
    instance = parse_psplib_mm(CORE_INSTANCE)

    assert instance.project_count == 1
    assert instance.declared_job_count == 8
    assert instance.horizon == 40
    assert instance.renewable_capacities == (4, 3)
    assert instance.nonrenewable_capacities == (12, 10)
    assert instance.source_job_ids == (1,)
    assert instance.sink_job_ids == (8,)
    assert instance.real_job_ids == (2, 3, 4, 5, 6, 7)
    assert len(instance.source_sha256) == 64

    activity_two = instance.activity_map[2]
    assert activity_two.predecessors == (1,)
    assert activity_two.successors == (4, 5)
    assert [mode.mode_id for mode in activity_two.modes] == [1, 2]
    assert activity_two.modes[1].duration == 2
    assert activity_two.modes[1].renewable_demands == (4, 1)
    assert activity_two.modes[1].nonrenewable_demands == (3, 2)
    assert instance.activity_map[5].predecessors == (2, 3)


def test_parser_fails_closed_for_doubly_constrained_and_cycles():
    with pytest.raises(MMRCPSPFormatError, match="doubly constrained"):
        parse_psplib_mm(DATA_ROOT / "mmrcpsp_unsupported_doubly.mm")

    cyclic = CORE_INSTANCE.read_text(encoding="utf-8").replace(
        "   8        1          0",
        "   8        1          1           1",
    )
    with pytest.raises(MMRCPSPFormatError, match="directed cycle"):
        parse_psplib_mm_text(cyclic, source_name="cycle.mm")


def test_feasible_actions_preserve_nonrenewable_budget_for_remaining_jobs():
    instance = parse_psplib_mm(RESERVE_INSTANCE)

    initial = feasible_mode_actions(
        instance,
        eligible_job_ids=(2, 3),
        unscheduled_job_ids=(2, 3, 4, 5),
        renewable_available=(2,),
        nonrenewable_consumed=(0,),
    )
    assert {(row.job_id, row.mode_id) for row in initial} == {
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
    }

    after_fast_job_two = feasible_mode_actions(
        instance,
        eligible_job_ids=(3,),
        unscheduled_job_ids=(3, 4, 5),
        renewable_available=(1,),
        nonrenewable_consumed=(3,),
    )
    assert [(row.job_id, row.mode_id) for row in after_fast_job_two] == [(3, 1)]
    assert after_fast_job_two[0].nonrenewable_reserve_after == (4,)


def test_nonrenewable_action_certificate_handles_cross_mode_coupling_exactly():
    instance = parse_psplib_mm(
        DATA_ROOT / "mmrcpsp_nonrenewable_coupled_infeasible.mm"
    )

    actions = feasible_mode_actions(
        instance,
        eligible_job_ids=(2, 3, 4),
        unscheduled_job_ids=(2, 3, 4, 5),
        renewable_available=(3,),
        nonrenewable_consumed=(0, 0),
    )

    # Per-coordinate minima are (0, 0), but no joint choice of three modes fits
    # capacities (4, 4).  The exact Pareto-frontier reserve must reject all rows.
    assert actions == ()


def test_infeasible_instance_fails_the_gate_without_fabricating_a_schedule():
    path = DATA_ROOT / "mmrcpsp_nonrenewable_coupled_infeasible.mm"

    report = build_mmrcpsp_benchmark((path,))

    assert report["gate"]["pass"] is False
    assert report["gate"]["all_policy_schedules_feasible"] is False
    for result in report["instances"][path.stem]["policy_results"].values():
        assert result["feasible"] is False
        assert result["failure_reason"].startswith("resource_deadlock:")
        assert result["metrics"]["missing_job_ids"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("policy", POLICY_ORDER)
def test_every_policy_builds_a_precedence_and_resource_feasible_schedule(policy):
    instance = parse_psplib_mm(CORE_INSTANCE)

    result = schedule_instance(instance, policy)

    assert result.feasible is True
    assert result.failure_reason is None
    assert result.metrics["schedule_feasible"] is True
    assert result.metrics["resource_violation_units"] == 0.0
    assert result.metrics["precedence_violation_count"] == 0
    assert result.metrics["missing_job_ids"] == []
    assert len(result.entries) == instance.declared_job_count
    assert result.metrics["makespan"] > 0
    assert result.metrics["mean_flow_time"] > 0
    assert result.metrics["renewable_configuration_l1"] >= 0


def test_benchmark_is_deterministic_and_does_not_gate_on_ours_dominance():
    paths = (CORE_INSTANCE, RESERVE_INSTANCE)

    first = build_mmrcpsp_benchmark(paths)
    second = build_mmrcpsp_benchmark(tuple(reversed(paths)))

    assert first == second
    assert first["gate"] == {
        "status": "MMRCPSP_CORE_FEASIBILITY_PASS",
        "pass": True,
        "all_policy_schedules_feasible": True,
        "zero_resource_violation": True,
        "zero_precedence_violation": True,
        "ours_dominance_not_required_for_pass": True,
    }
    assert first["policy_order"][0] == OURS_POLICY
    assert first["claim_boundary"]["strong_claim_ready"] is False
    assert "neither an exact solver" in first["claim_boundary"]["algorithm_scope"]
    assert "Fixture success" in first["claim_boundary"]["evidence_scope"]
    assert len(first["claim_boundary"]["strong_claim_blockers"]) == 3
    for instance_row in first["instances"].values():
        assert instance_row["ours_vs_baselines"][
            "dominance_is_descriptive_not_gate"
        ] is True


def test_json_and_markdown_artifacts_are_byte_deterministic(tmp_path):
    report = build_mmrcpsp_benchmark((CORE_INSTANCE, RESERVE_INSTANCE))
    json_path = tmp_path / "mmrcpsp.json"
    markdown_path = tmp_path / "mmrcpsp.md"

    write_artifacts(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    write_artifacts(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    parsed = json.loads(first_json)
    assert parsed["schema_version"] == "scheduleurm.mmrcpsp.benchmark.v1"
    assert b"Strong claim ready" in first_markdown
    assert b"MMRCPSP_CORE_FEASIBILITY_PASS" in first_markdown


def test_ours_trace_exposes_finite_mode_action_score_components():
    instance = parse_psplib_mm(CORE_INSTANCE)

    result = schedule_instance(instance, OURS_POLICY)
    snapshot = result.snapshot()

    assert snapshot["feasible"] is True
    assert all(row["action_id"] for row in snapshot["schedule"])
    assert snapshot["selection_semantics"] == (
        "finite_project_lower_service_minus_bounded_penalty"
    )
    assert all(row["queue_weight_proxy"] > 0 for row in snapshot["schedule"])
    assert all(row["lower_service"] >= 0 for row in snapshot["schedule"])
    assert all(row["penalty_units"] >= 0 for row in snapshot["schedule"])
    for row in snapshot["schedule"]:
        if row["duration"] > 0:
            assert row["action_score"] == pytest.approx(
                row["queue_weight_proxy"] * row["lower_service"]
                - row["penalty_units"],
                abs=1e-8,
            )
    assert any(
        row["action_type"] == "launch_reconfigured_mode"
        for row in snapshot["schedule"]
    )
