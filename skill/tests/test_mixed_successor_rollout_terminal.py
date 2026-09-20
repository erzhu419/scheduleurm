"""Hand-computed mixed-lane frames, temporary queues, and frozen fluid tails."""
from dataclasses import dataclass

import pytest

from algorithm.arrival_rollout_terminal import ArrivalPath, evaluate_arrival_rollout
from algorithm.batch_rollout_frame_candidates import _FrameCandidate, _build_rollout_profiles
from algorithm.mixed_successor_rollout_terminal import evaluate_mixed_successor_rollout


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


OPTIONS = (Option((1, 0), 2.0, ((1.0,), ())),
           Option((0, 1), 3.0, ((), (2.0,))))
DURATIONS = (2.0, 3.0)
PROFILES, _ = _build_rollout_profiles(((1, 0), (0, 1)), DURATIONS)
PATH = ArrivalPath(9.0, ((), ()), 0, 2)


def test_mixed_lanes_recompute_priority_and_use_max_duration_actual_offsets_and_original_tail():
    first = _FrameCandidate((0, None), (1, 0), 2.0, 3.0)
    args = (OPTIONS, (6, 5), first, 2, 1.0, (0.5, 0.5), PROFILES, DURATIONS, PATH)
    old = evaluate_arrival_rollout(*args)
    result = evaluate_mixed_successor_rollout(*args)
    assert result.steps[0] == old.steps[0]
    assert tuple(step["departures"] for step in old.steps[1:]) == ((2, 0), (0, 2))
    # At Q=(5,5), coordinate order picks A. Decrementing its temporary
    # queue makes B win the second lane; the same occurs at Q=(4,4).
    assert tuple(step["departures"] for step in result.steps[1:]) == ((1, 1), (1, 1))
    assert tuple(step["duration_s"] for step in result.steps[1:]) == (3.0, 3.0)
    assert tuple(step["queue_after"] for step in result.steps) == ((5, 5), (4, 4), (3, 3))
    # Each mixed frame has max duration3, offsets1+2, and respectively
    # 8 or 6 unserved jobs; neither duration sum nor barrier offsets apply.
    assert tuple(step["area"] for step in result.steps) == (21.0, 27.0, 21.0)
    assert result.current_area == 21.0 and result.successor_area == 48.0
    assert result.tail_area == pytest.approx(18.0)  # 3^2/(2*.5) in each class.
    assert result.cost == pytest.approx(87.0)
    assert old.cost == pytest.approx(79.0)
    assert result.attempts == 2
    assert result.lane_fill_evaluations == result.temporary_queue_updates == 4
    assert result.class_evaluations == 8 and result.profile_probes == 4
    # Three 2-coordinate boundaries, two tail coordinates, four decrements.
    assert result.queue_coordinate_evaluations == 12
    assert result.current_completion_offset_reads == 1 and result.completion_offset_reads == 4
    assert result.population_jobs == 11 and result.population_error == pytest.approx(0.0, abs=1e-12)
    assert result.horizon_excess == 0.0


def test_tie_after_first_lane_uses_coordinate_order_and_empty_lane_is_charged_once():
    first = _FrameCandidate((None, None), (0, 0), 1.0, 0.0)
    result = evaluate_mixed_successor_rollout(
        OPTIONS, (2, 1), first, 2, 1.0, (1.0, 1.0), PROFILES, DURATIONS, PATH)
    # First choose A at (2,1); the new tie at (1,1) chooses A again.
    # The second frame serves the sole B job, then visits one vacant lane.
    assert tuple(step["departures"] for step in result.steps[1:]) == ((2, 0), (0, 1))
    assert tuple(step["duration_s"] for step in result.steps[1:]) == (2.0, 3.0)
    assert tuple(step["area"] for step in result.steps) == (3.0, 4.0, 2.0)
    assert result.cost == 9.0 and result.tail_area == 0.0
    assert result.lane_fill_evaluations == 4 and result.temporary_queue_updates == 3
    assert result.class_evaluations == 5 and result.profile_probes == 3
    assert result.queue_coordinate_evaluations == 11
    assert result.attempts == 2 and result.population_error == 0.0
