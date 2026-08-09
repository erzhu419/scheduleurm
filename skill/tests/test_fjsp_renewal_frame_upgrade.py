from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_renewal_frame_upgrade import (
    build_fjsp_renewal_frame_schedule,
)


MK01 = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "fjsp_brandimarte_suite"
    / "mk01.txt"
)


@pytest.fixture(scope="module")
def report():
    return build_fjsp_renewal_frame_schedule(load_fjsp_instance(MK01))


def test_fjsp_renewal_family_is_feasible_and_exactly_maximized(report):
    family = report["candidate_family"]
    selection = report["selection"]

    assert family
    assert all(row["feasible"] for row in family)
    assert selection["exact_generated_family_argmax"] is True
    assert selection["generated_family_oracle_gap"] == 0
    assert selection["score_audit"]["renewal_score"] == max(
        row["renewal_score"] for row in family
    )


def test_fjsp_score_uses_physical_departures_and_variable_frame(report):
    score = report["selection"]["score_audit"]

    assert set(score["physical_departures"]["job_departure_indicators"]) == {1}
    assert score["holding_cost"]["holding_reward_is_not_physical_service"] is True
    assert score["frame"]["variable_duration_action"] is True
    assert score["frame"]["duration_tau"] == report["metrics"]["makespan"]


def test_fjsp_claim_boundary_does_not_infer_global_superiority(report):
    assert report["baseline_union_in_candidate_family"] is True
    assert report["performance_superiority_claim_ready"] is False
    assert report["theorem_mapping"]["global_fjsp_optimality_claimed"] is False
    assert report["theorem_mapping"]["stochastic_recurrence_claimed"] is False
