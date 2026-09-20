"""A shared deterministic-arrival path with two paid batch successor frames.

The path is a fixed quadrature surrogate, not a Poisson expectation or the
experiment's arrival tape. Both candidate evaluations use the same population
through a common horizon; fluid service after their two successors completes
every remaining job, including arrivals at the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArrivalPath:
    horizon: float
    times: tuple[tuple[float, ...], ...]
    event_generations: int
    rate_reads: int


@dataclass(frozen=True)
class ArrivalRolloutEvaluation:
    cost: float
    current_area: float
    successor_area: float
    tail_area: float
    attempts: int
    class_evaluations: int
    profile_probes: int
    completion_offset_reads: int
    current_completion_offset_reads: int
    arrival_event_reads: int
    tail_segments: int
    queue_coordinate_evaluations: int
    terminal_value_coordinate_evaluations: int
    population_jobs: int
    population_error: float
    horizon_excess: float
    steps: tuple[dict, ...]


def build_arrival_path(arrival_rates, horizon: float) -> ArrivalPath:
    """Generate each stored event once at (n - 0.5) / rate, including H."""
    times = []
    generated = rate_reads = 0
    for rate in arrival_rates:
        rate_reads += 1
        events = []
        if rate > 0.0:
            n = 1
            event = (n - 0.5) / rate
            while event <= horizon:
                events.append(event)
                generated += 1
                n += 1
                event = (n - 0.5) / rate
        times.append(tuple(events))
    return ArrivalPath(horizon, tuple(times), generated, rate_reads)


def fluid_interval(queue: float, service_rate: float, duration: float) -> tuple[float, float]:
    """Integrate reflected constant-rate clearing until the next arrival."""
    remaining = max(queue - service_rate * duration, 0.0)
    area = 0.5 * (queue + remaining) * min(duration, queue / service_rate)
    return area, remaining


def evaluate_arrival_rollout(options, queues, first_action, lanes, idle_duration,
                             service_rates, profiles, durations, arrival_path) -> ArrivalRolloutEvaluation:
    """Evaluate one fixed first action, two paid successors, then a fluid tail.

    first_action has indices, service and duration fields. Event reads include
    boundary lookahead peeks. Queue-coordinate counts cover each boundary
    update and each fluid-service or tail-arrival update. Tail-value counts
    cover scalar interval integrations and each class's final clearing value.
    """
    remaining = tuple(queues)
    now = 0.0
    cursors = [0] * len(remaining)
    event_reads = current_offset_reads = successor_offset_reads = 0
    queue_evaluations = class_evaluations = profile_probes = 0
    completed = 0.0
    steps = []

    def frame(indices, service, duration, is_current):
        nonlocal remaining, now, event_reads, current_offset_reads, successor_offset_reads
        nonlocal queue_evaluations, completed
        before = remaining
        residual = tuple(q - served for q, served in zip(before, service))
        end = now + duration
        offset_area = 0.0
        for index in indices:
            if index is not None:
                for offsets in options[index].completion_offsets:
                    for offset in offsets:
                        offset_area += offset
                        if is_current:
                            current_offset_reads += 1
                        else:
                            successor_offset_reads += 1
        arrivals = [0] * len(remaining)
        barrier_area = 0.0
        for coordinate, events in enumerate(arrival_path.times):
            cursor = cursors[coordinate]
            while cursor < len(events):
                event = events[cursor]
                event_reads += 1
                if event > end:
                    break
                arrivals[coordinate] += 1
                barrier_area += end - event
                cursor += 1
            cursors[coordinate] = cursor
        area = sum(residual) * duration + offset_area + barrier_area
        remaining = tuple(q + arrived for q, arrived in zip(residual, arrivals))
        queue_evaluations += len(remaining)
        completed += sum(service)
        steps.append({
            "start_s": now, "duration_s": duration, "queue_before": before,
            "departures": tuple(service), "arrivals": tuple(arrivals),
            "queue_after": remaining, "area": area,
        })
        now = end
        return area

    current_area = frame(first_action.indices, first_action.service, first_action.duration, True)
    successor_area = 0.0
    for _ in range(2):
        chosen_class = None
        best_priority = -1.0
        for coordinate, (count, rate) in enumerate(zip(remaining, service_rates)):
            if count:
                class_evaluations += 1
                priority = count / rate
                if priority > best_priority:
                    chosen_class, best_priority = coordinate, priority
        service = [0] * len(remaining)
        indices = (None,) * lanes
        duration = idle_duration
        if chosen_class is not None:
            chosen_profile = None
            for batch, index in profiles[chosen_class]:
                profile_probes += 1
                if batch <= remaining[chosen_class]:
                    chosen_profile = (batch, index)
                    break
            if chosen_profile is None:
                raise ValueError("rollout requires a fitting pure profile for every nonempty chosen class")
            batch, index = chosen_profile
            copies = min(lanes, remaining[chosen_class] // batch)
            service[chosen_class] = copies * batch
            indices = (index,) * copies + (None,) * (lanes - copies)
            duration = durations[index]
        if any(served > count for served, count in zip(service, remaining)):
            raise ValueError("constructed rollout frame exceeds its visible queue")
        successor_area += frame(indices, service, duration, False)

    tail_area = 0.0
    tail_segments = terminal_evaluations = 0
    for coordinate, (count, rate) in enumerate(zip(remaining, service_rates)):
        queue = float(count)
        time = now
        events = arrival_path.times[coordinate]
        for cursor in range(cursors[coordinate], len(events)):
            event = events[cursor]
            event_reads += 1
            area, after_service = fluid_interval(queue, rate, event - time)
            tail_area += area
            completed += queue - after_service
            queue = after_service + 1.0
            queue_evaluations += 2
            tail_segments += 1
            terminal_evaluations += 1
            time = event
        area, after_service = fluid_interval(queue, rate, max(0.0, arrival_path.horizon - time))
        tail_area += area + after_service**2 / (2.0 * rate)
        completed += queue - after_service + after_service
        queue_evaluations += 1
        tail_segments += 1
        terminal_evaluations += 2
    population = sum(queues) + arrival_path.event_generations
    return ArrivalRolloutEvaluation(
        cost=current_area + successor_area + tail_area,
        current_area=current_area, successor_area=successor_area, tail_area=tail_area,
        attempts=2, class_evaluations=class_evaluations, profile_probes=profile_probes,
        completion_offset_reads=successor_offset_reads,
        current_completion_offset_reads=current_offset_reads,
        arrival_event_reads=event_reads, tail_segments=tail_segments,
        queue_coordinate_evaluations=queue_evaluations,
        terminal_value_coordinate_evaluations=terminal_evaluations,
        population_jobs=population, population_error=abs(population - completed),
        horizon_excess=max(0.0, now - arrival_path.horizon), steps=tuple(steps),
    )
