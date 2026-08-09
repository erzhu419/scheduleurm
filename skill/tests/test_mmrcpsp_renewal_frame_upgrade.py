from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm
from algorithm.experiments.mmrcpsp_renewal_frame_upgrade import (
    build_mmrcpsp_renewal_frame_schedule,
)


INSTANCE = Path(__file__).resolve().parents[2] / "tests" / "data" / "mmrcpsp_core.mm"


@pytest.fixture(scope="module")
def report():
    return build_mmrcpsp_renewal_frame_schedule(parse_psplib_mm(INSTANCE))


def test_mmrcpsp_renewal_family_is_feasible_and_exactly_maximized(report):
    family = report["candidate_family"]
    selection = report["selection"]

    assert family
    assert all(row["feasible"] for row in family)
    assert selection["exact_generated_family_argmax"] is True
    assert selection["generated_family_oracle_gap"] == 0
    assert selection["score_audit"]["renewal_score"] == max(
        row["renewal_score"] for row in family
    )


def test_mmrcpsp_uses_departures_not_holding_reward_as_service(report):
    score = report["selection"]["score_audit"]

    assert set(score["physical_departures"]["job_departure_indicators"]) == {1}
    assert score["physical_departures"]["project_departure_indicator"] == 1
    assert score["holding_cost"]["holding_reward_is_not_physical_service"] is True
    assert score["frame"]["variable_duration_action"] is True
    assert score["frame"]["duration_tau"] == report["metrics"]["makespan"]


def test_mmrcpsp_configuration_is_not_mislabeled_as_service(report):
    diagnostics = report["selection"]["configuration_diagnostics"]

    assert diagnostics["excluded_from_physical_service"] is True
    assert report["theorem_mapping"][
        "mode_and_resource_configuration_are_diagnostics_not_service"
    ] is True


def test_mmrcpsp_claim_boundary_does_not_infer_global_superiority(report):
    assert report["baseline_union_in_candidate_family"] is True
    assert report["performance_superiority_claim_ready"] is False
    assert report["theorem_mapping"]["global_mmrcpsp_optimality_claimed"] is False
    assert report["theorem_mapping"]["stochastic_recurrence_claimed"] is False
