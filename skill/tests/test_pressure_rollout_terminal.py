"""Independent costs for the successor-priority change with the frozen tail."""
from dataclasses import dataclass

import pytest

from algorithm.arrival_rollout_terminal import ArrivalPath, evaluate_arrival_rollout
from algorithm.batch_rollout_frame_candidates import _FrameCandidate, _build_rollout_profiles
from algorithm.pressure_rollout_terminal import evaluate_pressure_rollout, static_pressure_rates


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def test_static_rates_use_effective_duration_and_only_retained_pure_profiles():
    # Sub-idle duration must not create a 10 jobs/s single-job rate. The
    # duplicate slow single and the fast mixed action are not pure choices.
    requirements = ((1, 0), (3, 0), (1, 0), (0, 1), (2, 2))
    reads = []

    class Durations(tuple):
        def __getitem__(self, index):
            reads.append(index)
            return super().__getitem__(index)

    durations = Durations(max(1.0, duration) for duration in (0.2, 4.0, 3.0, 2.0, 0.1))
    profiles, _ = _build_rollout_profiles(requirements, durations)
    reads.clear()
    rates, profile_reads, rate_evaluations = static_pressure_rates(profiles, durations, 2)
    assert rates == pytest.approx((2.0, 1.0))
    assert profiles == (((3, 1), (1, 0)), ((1, 3),))
    assert reads == [1, 0, 3]
    assert profile_reads == rate_evaluations == len(reads) == 3


def test_priority_reverses_successors_but_keeps_first_frame_arrivals_and_fluid_mu():
    options = (Option((1, 0), 2.0, ((1.0,), ())),
               Option((0, 1), 1.0, ((), (1.0,))))
    durations = (2.0, 1.0)
    profiles, _ = _build_rollout_profiles(((1, 0), (0, 1)), durations)
    rates, _, _ = static_pressure_rates(profiles, durations, 1)
    first = _FrameCandidate((0,), (1, 0), 2.0, 2.5)
    path = ArrivalPath(6.0, ((0.5,), (3.0,)), 2, 2)
    args = (options, (5, 4), first, 1, 1.0, (0.1, 1.0), profiles, durations, path)
    old = evaluate_arrival_rollout(*args)
    result = evaluate_pressure_rollout(*args, rates)

    # The first frame serves A; its residual jobs wait 16 job-seconds, the
    # completion contributes 1, and the arrival at .5 contributes 1.5.
    assert result.steps[0] == old.steps[0]
    assert result.current_area == 18.5
    assert tuple(step["departures"] for step in old.steps[1:]) == ((1, 0), (1, 0))
    assert tuple(step["departures"] for step in result.steps[1:]) == ((0, 1), (0, 1))
    assert tuple(step["area"] for step in result.steps[1:]) == (9.0, 9.0)
    assert result.steps[-1]["queue_after"] == (5, 3)
    # At t=4 all path arrivals have occurred. Continued service at original
    # mu, including clearing after H, gives 5^2/(2*.1) + 3^2/(2*1) = 129.5.
    assert result.successor_area == 18.0
    assert result.tail_area == pytest.approx(129.5)
    assert result.cost == pytest.approx(166.0)
    assert old.cost == pytest.approx(111.0)
    assert result.attempts == old.attempts == 2
    assert result.class_evaluations == 4 and result.profile_probes == 2
    assert result.current_completion_offset_reads == 1
    assert result.completion_offset_reads == 2
    assert result.population_jobs == old.population_jobs == 11
    assert result.population_error == old.population_error == pytest.approx(0.0, abs=1e-12)
    assert result.horizon_excess == old.horizon_excess == 0.0
