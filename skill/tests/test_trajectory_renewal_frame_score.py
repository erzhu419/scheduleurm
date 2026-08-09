from __future__ import annotations

import pytest

from algorithm.experiments.trajectory_renewal_frame_score import (
    RenewalFrameScoreError,
    complete_trajectory_renewal_score,
)


def _score(completion, makespan):
    return complete_trajectory_renewal_score(
        completion_times=completion,
        frame_duration=makespan,
        common_horizon_bound=20,
        project_queue_weight=len(completion),
    )


def test_physical_departures_are_separate_from_holding_reward():
    report = _score((4, 7), 7)

    assert report["physical_departures"]["job_departure_indicators"] == [1, 1]
    assert report["physical_departures"]["project_departure_indicator"] == 1
    assert report["holding_cost"]["holding_reward_coordinates"] == [0.8, 0.65]
    assert report["holding_cost"]["holding_reward_is_not_physical_service"] is True


def test_renewal_score_strictly_prefers_pareto_improvement():
    slower = _score((6, 10), 10)
    faster_makespan = _score((6, 9), 9)
    faster_flow = _score((5, 10), 10)

    assert faster_makespan["renewal_score"] > slower["renewal_score"]
    assert faster_flow["renewal_score"] > slower["renewal_score"]


def test_bounded_extra_penalty_is_audited():
    report = complete_trajectory_renewal_score(
        completion_times=(4, 7),
        frame_duration=7,
        common_horizon_bound=20,
        extra_penalty=0.02,
        extra_penalty_bound=0.03,
    )

    assert report["penalty"]["extra_penalty"] == 0.02
    assert report["penalty"]["uniform_bound"] == 0.08
    with pytest.raises(RenewalFrameScoreError, match="declared bound"):
        complete_trajectory_renewal_score(
            completion_times=(4, 7),
            frame_duration=7,
            common_horizon_bound=20,
            extra_penalty=0.04,
            extra_penalty_bound=0.03,
        )


def test_invalid_frame_fails_closed():
    with pytest.raises(RenewalFrameScoreError, match=r"\(0, H\]"):
        complete_trajectory_renewal_score(
            completion_times=(4, 7),
            frame_duration=21,
            common_horizon_bound=20,
        )
