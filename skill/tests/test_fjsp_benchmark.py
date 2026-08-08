from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from algorithm.experiments.fjsp_benchmark import (
    POLICY_ORDER,
    ReassignmentConfig,
    ScheduledOperation,
    artifact_json,
    build_schedule,
    enumerate_feasible_actions,
    markdown_report,
    run_benchmark,
    verify_schedule,
    write_artifacts,
)
from algorithm.experiments.fjsp_instances import (
    FJSPParseError,
    load_fjsp_instance,
    parse_fjsp_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "tests" / "data"
INSTANCE_PATH = DATA_ROOT / "fjsp_brandimarte_tiny.fjs"
MK01_PATH = DATA_ROOT / "fjsp_brandimarte_mk01.fjs"


def test_parse_common_brandimarte_format_and_normalize_one_based_machines():
    instance = load_fjsp_instance(INSTANCE_PATH)

    assert instance.job_count == 4
    assert instance.machine_count == 3
    assert instance.operation_count == 8
    assert instance.source_machine_index_base == 1
    assert instance.header_metadata == ("2",)
    assert [
        (option.machine_id, option.processing_time)
        for option in instance.operation(0, 0).alternatives
    ] == [(0, 2), (1, 5)]
    assert instance.operation(2, 0).preferred_machine_id == 0
    assert instance.lower_bound_makespan == 6


def test_parser_accepts_wrapped_zero_based_records_and_comments():
    instance = parse_fjsp_text(
        """
        // one job, two machines
        1 2 1.5
        2
        1 0 3
        2 0 2 1 4  # operation two
        """,
        name="zero_based",
    )

    assert instance.source_machine_index_base == 0
    assert instance.header_metadata == ("1.5",)
    assert instance.operation_count == 2
    assert [option.machine_id for option in instance.operation(0, 1).alternatives] == [0, 1]


def test_official_brandimarte_mk01_parses_and_ours_is_feasible_nondominated_dispatch():
    instance = load_fjsp_instance(MK01_PATH)
    report = run_benchmark(instance, known_best_makespan=40)

    assert instance.job_count == 10
    assert instance.machine_count == 6
    assert instance.operation_count == 55
    assert report["gate"]["pass"] is True
    assert report["policies"]["ours_robust_maxweight"]["metrics"]["makespan"] == 45
    assert report["policies"]["ours_robust_maxweight"]["metrics"][
        "makespan_over_known_best"
    ] == 1.125
    assert report["comparison"]["ours_pareto_dominated_by_baselines"] == []
    assert artifact_json(report) == (
        DATA_ROOT / "fjsp_brandimarte_mk01_benchmark.json"
    ).read_text(encoding="utf-8")
    assert markdown_report(report) == (
        DATA_ROOT / "fjsp_brandimarte_mk01_benchmark.md"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "text, message",
    [
        ("1 2\n1 1 3 4\n", "outside detected"),
        ("1 2\n1 2 1 3 1 4\n", "repeats machine"),
        ("1 2\n1 1 1\n", "truncated instance"),
        ("1 2\n1 1 1 3 99\n", "unexpected trailing"),
        ("1 2\n0\n", "at least one operation"),
    ],
)
def test_parser_fails_closed_on_malformed_instances(text: str, message: str):
    with pytest.raises(FJSPParseError, match=message):
        parse_fjsp_text(text)


def test_feasible_actions_are_next_operations_cross_eligible_machines_only():
    instance = load_fjsp_instance(INSTANCE_PATH)
    actions = enumerate_feasible_actions(
        instance,
        next_operation_by_job=[0, 0, 0, 0],
        job_ready_times=[0.0, 0.0, 0.0, 0.0],
        machine_timelines=[[], [], []],
    )

    assert len(actions) == 9
    assert {(row.operation.job_id, row.operation.operation_id) for row in actions} == {
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    }
    assert all(row.start_time == 0.0 for row in actions)
    assert all(not row.was_reassigned for row in actions)


def test_insertion_action_respects_predecessor_and_existing_machine_intervals():
    instance = load_fjsp_instance(INSTANCE_PATH)
    occupied = ScheduledOperation(
        job_id=1,
        operation_id=0,
        machine_id=1,
        start_time=3.0,
        end_time=7.0,
        processing_time=4.0,
        reconfiguration_time=0.0,
        preferred_machine_id=1,
        reassignment_semantics_enabled=False,
    )
    actions = enumerate_feasible_actions(
        instance,
        next_operation_by_job=[1, 0, 0, 0],
        job_ready_times=[2.0, 0.0, 0.0, 0.0],
        machine_timelines=[[], [occupied], []],
    )
    job_zero = [row for row in actions if row.operation.job_id == 0]

    assert {row.operation.operation_id for row in job_zero} == {1}
    by_machine = {row.machine_id: row for row in job_zero}
    assert by_machine[1].start_time == 7.0
    assert by_machine[2].start_time == 2.0


def test_every_policy_builds_a_complete_feasible_schedule_and_results_differ():
    instance = load_fjsp_instance(INSTANCE_PATH)
    config = ReassignmentConfig(enabled=True, setup_time=1.0)
    results = {
        policy: build_schedule(instance, policy, reassignment=config)
        for policy in POLICY_ORDER
    }

    assert all(result["feasibility"]["pass"] for result in results.values())
    assert all(len(result["schedule"]) == instance.operation_count for result in results.values())
    assert len({result["metrics"]["makespan"] for result in results.values()}) >= 3
    assert any(result["metrics"]["reconfiguration_count"] > 0 for result in results.values())


def test_ours_uses_exact_theorem_dispatch_candidate_oracle():
    instance = load_fjsp_instance(INSTANCE_PATH)
    result = build_schedule(
        instance,
        "ours_robust_maxweight",
        reassignment=ReassignmentConfig(enabled=True, setup_time=1.0),
    )

    assert len(result["decision_trace"]) == instance.operation_count
    for decision in result["decision_trace"]:
        audit = decision["selection_audit"]
        selected = audit["selected_global_action"]
        assert audit["rule"] == "robust_maxweight_lower_service"
        assert selected["score_semantics"] == "robust_maxweight_lower_service"
        assert selected["oracle_gap_alpha0"] == 0.0
        assert selected["oracle_gap_alpha1"] == 0.0
        assert selected["candidate_configuration_count"] == decision["candidate_count"]


def test_reassignment_is_opt_in_and_never_claims_running_operation_migration():
    instance = load_fjsp_instance(INSTANCE_PATH)
    disabled = run_benchmark(instance)
    enabled = run_benchmark(
        instance,
        reassignment=ReassignmentConfig(enabled=True, setup_time=1.0),
    )

    assert disabled["semantics"]["reassignment"]["scope"] == "disabled"
    assert all(
        result["metrics"]["reconfiguration_count"] == 0
        and result["metrics"]["reconfiguration_time"] == 0
        for result in disabled["policies"].values()
    )
    assert enabled["semantics"]["reassignment"]["scope"] == "unstarted_operation_only"
    assert enabled["semantics"]["reassignment"]["running_operation_migration_supported"] is False
    assert enabled["gate"]["running_operation_migration_claimed"] is False
    assert any(
        result["metrics"]["reconfiguration_count"] > 0
        for result in enabled["policies"].values()
    )


def test_feasibility_certificate_detects_machine_overlap():
    instance = load_fjsp_instance(INSTANCE_PATH)
    result = build_schedule(instance, "earliest_completion_time")
    schedule = [ScheduledOperation(**{
        "job_id": row["job_id"],
        "operation_id": row["operation_id"],
        "machine_id": row["machine_id"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "processing_time": row["processing_time"],
        "reconfiguration_time": row["reconfiguration_time"],
        "preferred_machine_id": row["preferred_machine_id"],
        "reassignment_semantics_enabled": False,
    }) for row in result["schedule"]]
    same_machine = next(
        (left, right)
        for index, left in enumerate(schedule)
        for right in schedule[index + 1 :]
        if left.machine_id == right.machine_id
    )
    left, right = same_machine
    corrupted = [
        replace(
            item,
            start_time=left.start_time,
            end_time=left.start_time + item.occupied_time,
        )
        if item == right
        else item
        for item in schedule
    ]

    certificate = verify_schedule(instance, corrupted)

    assert certificate["pass"] is False
    assert certificate["machine_nonoverlap"] is False
    assert any("overlap" in error for error in certificate["errors"])


def test_report_is_deterministic_and_matches_checked_in_artifacts(tmp_path: Path):
    instance = load_fjsp_instance(INSTANCE_PATH)
    config = ReassignmentConfig(enabled=True, setup_time=1.0)
    first = run_benchmark(instance, reassignment=config)
    second = run_benchmark(instance, reassignment=config)

    assert artifact_json(first) == artifact_json(second)
    assert "timestamp" not in artifact_json(first).lower()
    assert artifact_json(first) == (
        DATA_ROOT / "fjsp_brandimarte_tiny_benchmark.json"
    ).read_text(encoding="utf-8")
    assert markdown_report(first) == (
        DATA_ROOT / "fjsp_brandimarte_tiny_benchmark.md"
    ).read_text(encoding="utf-8")

    json_path, markdown_path = write_artifacts(
        first,
        json_path=tmp_path / "result.json",
        markdown_path=tmp_path / "result.md",
    )
    assert json_path.read_text(encoding="utf-8") == artifact_json(first)
    assert markdown_path.read_text(encoding="utf-8") == markdown_report(first)
