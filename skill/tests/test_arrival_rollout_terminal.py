"""Hand calculations for one shared arrival path and its fluid continuation."""
from dataclasses import dataclass, replace

import pytest

from algorithm.arrival_rollout_terminal import build_arrival_path, evaluate_arrival_rollout, fluid_interval
from algorithm.batch_rollout_frame_candidates import _FrameCandidate, _build_rollout_profiles, _rollout_old_jobs


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float
    completion_offsets: tuple[tuple[float, ...], ...]


def _evaluate(options, queues, index, arrival_rates, service_rates):
    durations = tuple(max(1.0, option.duration) for option in options)
    profiles, _ = _build_rollout_profiles(tuple(option.requirements for option in options), durations)
    first = _FrameCandidate((index,), options[index].requirements, durations[index],
                            sum(q * d for q, d in zip(queues, options[index].requirements)) / durations[index])
    path = build_arrival_path(arrival_rates, 3 * max(durations))
    result = evaluate_arrival_rollout(options, queues, first, 1, 1.0, service_rates,
                                     profiles, durations, path)
    return result, path, first, profiles, durations


def test_midpoint_arrivals_have_one_absolute_origin_and_keep_fractional_carry():
    path = build_arrival_path((0.6, 0.0), 5.0)
    assert path.times[0] == pytest.approx((5 / 6, 2.5, 25 / 6))
    assert path.times[1] == ()
    assert path.horizon == 5.0
    assert path.event_generations == 3 and path.rate_reads == 2
    first = sum(0.0 < t <= 1.0 for t in path.times[0])
    second = sum(1.0 < t <= 2.0 for t in path.times[0])
    assert (first, second) == (1, 0)
    assert first + second == sum(t <= 2.0 for t in path.times[0])
    # Resetting the same midpoint phase each frame would invent a second job.
    assert first + first != first + second


def test_arrivals_cross_batch_threshold_only_at_boundaries_and_offsets_set_area():
    options = (Option((1,), 2.0, ((0.5,),)), Option((2,), 3.0, ((1.0, 3.0),)))
    result, path, *_ = _evaluate(options, (2,), 0, (0.5,), (1.0,))
    assert path.times == ((1.0, 3.0, 5.0, 7.0, 9.0),)
    # At t=2 the old residual plus the t=1 arrival makes batch 2 feasible.
    # Arrivals at t=3 and t=5 cannot join the batch dispatched at t=2.
    expected = (
        (0.0, 2.0, (2,), (1,), (1,), (2,), 3.5),
        (2.0, 3.0, (2,), (2,), (2,), (2,), 6.0),
        (5.0, 3.0, (2,), (2,), (1,), (1,), 5.0),
    )
    for step, row in zip(result.steps, expected, strict=True):
        actual = tuple(step[key] for key in ("start_s", "duration_s", "queue_before", "departures",
                                            "arrivals", "queue_after", "area"))
        assert actual == row
    assert result.current_area == 3.5 and result.successor_area == 11.0
    # Fluid clears q=1 on [8,9]; arrival exactly at H=9 has a 1/2 tail.
    assert result.tail_area == pytest.approx(1.0)
    assert result.cost == pytest.approx(15.5)
    assert result.attempts == 2 and result.profile_probes == 2
    assert result.completion_offset_reads == 4 and result.current_completion_offset_reads == 1
    assert result.population_jobs == 7
    assert result.population_error == pytest.approx(0.0, abs=1e-12)
    assert result.horizon_excess == 0.0


def test_fluid_integration_reflects_at_zero_and_preserves_clearance_value():
    assert fluid_interval(2.0, 1.0, 1.0) == (1.5, 1.0)
    assert fluid_interval(1.0, 2.0, 2.0) == (0.25, 0.0)
    assert fluid_interval(0.0, 2.0, 5.0) == (0.0, 0.0)
    for queue, rate, duration in ((2.0, 1.0, 0.25), (2.0, 1.0, 7.0), (4.0, 0.25, 2.0)):
        area, remaining = fluid_interval(queue, rate, duration)
        assert area + remaining**2 / (2 * rate) == pytest.approx(queue**2 / (2 * rate))


def test_different_first_frame_lengths_evaluate_the_same_population_until_common_horizon():
    options = (Option((1,), 1.0, ((1.0,),)), Option((2,), 3.0, ((3.0, 3.0),)))
    short, short_path, *_ = _evaluate(options, (2,), 0, (0.5,), (1.0,))
    long, long_path, *_ = _evaluate(options, (2,), 1, (0.5,), (1.0,))
    assert short_path == long_path
    assert short_path.horizon == 9.0
    assert short.population_jobs == long.population_jobs == 7  # two initial + five arrivals.
    assert short.steps[-1]["start_s"] + short.steps[-1]["duration_s"] == 5.0
    assert long.steps[-1]["start_s"] + long.steps[-1]["duration_s"] == 7.0
    # Short: frame areas 2+7+1; tail includes arrivals at 7 and 9, hence 1.5.
    # Long: frame areas 8+7+1; tail still includes arrival at 9, hence 1.
    assert (short.current_area, short.successor_area, short.tail_area) == pytest.approx((2.0, 8.0, 1.5))
    assert (long.current_area, long.successor_area, long.tail_area) == pytest.approx((8.0, 8.0, 1.0))
    assert short.cost == pytest.approx(11.5) and long.cost == pytest.approx(17.0)
    assert short.population_error == long.population_error == pytest.approx(0.0, abs=1e-12)


def test_zero_arrivals_reduce_to_the_frozen_two_step_batch_cost():
    options = (Option((1,), 1.0, ((1.0,),)), Option((2,), 3.0, ((3.0, 3.0),)))
    for queues, index, rate in (((2,), 0, 1.0), ((2,), 1, 1.0), ((100,), 0, 0.1)):
        result, path, first, profiles, durations = _evaluate(options, queues, index, (0.0,), (rate,))
        residual = tuple(q - d for q, d in zip(queues, first.service))
        old = _rollout_old_jobs(options, residual, 1, 1.0, (rate,), profiles, durations)
        current = (sum(queues) - sum(first.service)) * first.duration + sum(options[index].completion_offsets[0])
        expected = current + old.holding_area + old.tail_fluid_value
        assert result.cost == pytest.approx(expected)
        assert path.times == ((),) and path.event_generations == 0
        assert result.population_jobs == sum(queues)
        assert result.attempts == old.attempts == 2


def test_event_peeks_and_completion_offset_reads_are_charged_as_executed():
    reads = {"events": 0, "offsets": 0}

    class Events(tuple):
        def __getitem__(self, index):
            reads["events"] += 1
            return super().__getitem__(index)

    class Offsets(tuple):
        def __iter__(self):
            for value in super().__iter__():
                reads["offsets"] += 1
                yield value

    options = (Option((1,), 2.0, (Offsets((0.5,)),)),
               Option((2,), 3.0, (Offsets((1.0, 3.0)),)))
    path = build_arrival_path((0.5,), 9.0)
    path = replace(path, times=(Events(path.times[0]),))
    profiles, _ = _build_rollout_profiles(((1,), (2,)), (2.0, 3.0))
    first = _FrameCandidate((0,), (1,), 2.0, 1.0)
    result = evaluate_arrival_rollout(options, (2,), first, 1, 1.0, (1.0,), profiles, (2.0, 3.0), path)
    assert result.cost == pytest.approx(15.5)
    # Frame event reads: 2 + 3 + 2 (including next-event peeks), tail: 1.
    assert result.arrival_event_reads == reads["events"] == 8
    assert result.current_completion_offset_reads + result.completion_offset_reads == reads["offsets"] == 5
    assert result.queue_coordinate_evaluations == 6
    assert result.tail_segments == 2 and result.terminal_value_coordinate_evaluations == 3
