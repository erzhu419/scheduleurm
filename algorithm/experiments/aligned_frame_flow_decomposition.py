"""Exact flow-area accounting for completed synchronous-frame populations.

Dispatch-to-completion offsets include every phase in a macro action. Waiting
areas are attributed to the frame in which the waiting occurs, and grouped by
that frame's action multiset; they do not identify causal action effects.
"""
from __future__ import annotations

from collections import Counter, deque
import math
from typing import Any, Mapping, Sequence


COMPONENTS = ("deferred_visible", "new_arrival_barrier", "dispatch_to_completion")
MEAN_FIELDS = (
    "mean_deferred_visible_wait_s", "mean_arrival_barrier_wait_s",
    "mean_dispatch_to_completion_s",
)


def _population_summary(count: int, completed: int, areas: Sequence[float], direct_flow: float) -> dict:
    total = sum(areas)
    if count != completed:
        raise ValueError(f"flow decomposition requires a complete population: {completed}/{count}")
    if not math.isclose(total, direct_flow, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"flow-area identity does not match reconstructed completions: {total} != {direct_flow}")
    return {
        "job_count": count,
        "completed_jobs": completed,
        "flow_area_job_seconds": total,
        "flow_area_components": dict(zip(COMPONENTS, areas, strict=True)),
        "mean_flow_s": total / count if count else None,
        **{
            name: area / count if count else None
            for name, area in zip(MEAN_FIELDS, areas, strict=True)
        },
        "reconstructed_completion_flow_area_job_seconds": direct_flow,
        "flow_identity_error_job_seconds": total - direct_flow,
    }


def decompose_flow(
    model: Any,
    arrivals: Sequence[tuple[float, int, int]],
    frames: Sequence[Mapping[str, Any]],
    arrival_horizon: float,
) -> dict:
    """Reconstruct FIFO completion sums and the three disjoint flow components."""
    dimension = len(model.workloads)
    queues = [deque() for _ in range(dimension)]
    tape = sorted(arrivals)
    if not frames:
        raise ValueError("a complete frame trace is required")
    if any(time < 0.0 or time > arrival_horizon or not 0 <= coordinate < dimension
           for time, coordinate, _ in tape):
        raise ValueError("arrival tape lies outside the registered horizon or workload coordinates")
    population = [0] * dimension
    for _, coordinate, _ in tape:
        population[coordinate] += 1
    completed = [0] * dimension
    direct_flow = [0.0] * dimension
    areas = [[0.0] * len(COMPONENTS) for _ in range(dimension)]
    arrival_window_areas = [[0.0] * len(COMPONENTS) for _ in range(dimension)]
    pointer = 0
    elapsed = duration_sum = squared_duration_sum = 0.0
    occupied_seconds = unassigned_seconds = finished_barrier_seconds = 0.0
    loaded_frames = 0
    groups: dict[tuple, dict] = {}

    for frame_number, frame in enumerate(frames):
        start = float(frame["start_s"])
        duration = float(frame["duration_s"])
        if duration <= 0.0 or not math.isclose(start, elapsed, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"frame {frame_number}: frames must be positive and contiguous from time zero")
        end = start + duration
        while pointer < len(tape) and tape[pointer][0] <= start:
            time, coordinate, index = tape[pointer]
            queues[coordinate].append((time, index))
            pointer += 1
        before = tuple(len(queue) for queue in queues)
        if tuple(frame["queue_before"]) != before:
            raise ValueError(f"frame {frame_number}: queue_before disagrees with the arrival tape")
        indices = tuple(frame["option_indices"])
        if len(indices) != model.lanes:
            raise ValueError(f"frame {frame_number}: lane assignment count disagrees with the model")
        options = [model.options[index] for index in indices if index is not None]
        expected_duration = max((1.0, *(option.duration for option in options)))
        if not math.isclose(duration, expected_duration, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"frame {frame_number}: duration differs from the synchronized action duration")
        departures = tuple(sum(option.requirements[i] for option in options) for i in range(dimension))
        if tuple(frame["departures"]) != departures or any(d > q for q, d in zip(before, departures, strict=True)):
            raise ValueError(f"frame {frame_number}: departures disagree with feasible action requirements")
        frame_areas = [[(q - d) * duration, 0.0, 0.0] for q, d in zip(before, departures, strict=True)]
        window_exposure = max(0.0, min(end, arrival_horizon) - start)
        frame_window_areas = [[(q - d) * window_exposure, 0.0, 0.0] for q, d in zip(before, departures, strict=True)]
        actual_departures = [0] * dimension
        for option in options:
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    if offset < 0.0 or offset > option.duration:
                        raise ValueError(f"frame {frame_number}: completion offset lies outside its action")
                    arrival, _ = queues[coordinate].popleft()
                    frame_areas[coordinate][2] += offset
                    frame_window_areas[coordinate][2] += max(0.0, min(start + offset, arrival_horizon) - start)
                    direct_flow[coordinate] += start + offset - arrival
                    completed[coordinate] += 1
                    actual_departures[coordinate] += 1
        if tuple(actual_departures) != departures:
            raise ValueError(f"frame {frame_number}: completion offsets do not complete the declared departures")
        while pointer < len(tape) and tape[pointer][0] <= end:
            time, coordinate, index = tape[pointer]
            queues[coordinate].append((time, index))
            frame_areas[coordinate][1] += end - time
            frame_window_areas[coordinate][1] += max(0.0, min(end, arrival_horizon) - time)
            pointer += 1
        for coordinate in range(dimension):
            for component in range(len(COMPONENTS)):
                areas[coordinate][component] += frame_areas[coordinate][component]
                arrival_window_areas[coordinate][component] += frame_window_areas[coordinate][component]
        frame_totals = [sum(row[i] for row in frame_areas) for i in range(len(COMPONENTS))]

        key = tuple(sorted((model.options[index].action_id if index is not None else None for index in indices), key=lambda value: value or ""))
        if key not in groups:
            groups[key] = {
                "action_multiset": [{"action_id": action_id, "count": count} for action_id, count in Counter(key).items()],
                "frame_count": 0, "total_duration_s": 0.0,
                "flow_area_components": dict.fromkeys(COMPONENTS, 0.0),
            }
        group = groups[key]
        group["frame_count"] += 1
        group["total_duration_s"] += duration
        for name, value in zip(COMPONENTS, frame_totals, strict=True):
            group["flow_area_components"][name] += value
        occupied_seconds += sum(option.duration for option in options)
        unassigned_seconds += (model.lanes - len(options)) * duration
        finished_barrier_seconds += sum(duration - option.duration for option in options)
        loaded_frames += int(any(option.loaded for option in options))
        duration_sum += duration
        squared_duration_sum += duration * duration
        elapsed = end

    if pointer != len(tape) or any(queues):
        raise ValueError("frame trace ends before the complete arrival population is drained")
    if elapsed < arrival_horizon and not math.isclose(elapsed, arrival_horizon, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("frame trace does not cover the registered arrival horizon")
    total_areas = [sum(row[i] for row in areas) for i in range(len(COMPONENTS))]
    summary = _population_summary(len(tape), sum(completed), total_areas, sum(direct_flow))
    window_totals = [sum(row[i] for row in arrival_window_areas) for i in range(len(COMPONENTS))]
    drain_totals = [total - window for total, window in zip(total_areas, window_totals, strict=True)]
    return {
        "status": "PASS",
        "arrival_horizon_s": arrival_horizon,
        "elapsed_including_drain_s": elapsed,
        "frame_count": len(frames),
        **summary,
        "by_class": {
            workload: _population_summary(population[i], completed[i], areas[i], direct_flow[i])
            for i, workload in enumerate(model.workloads)
        },
        "frame_duration_mean_s": duration_sum / len(frames),
        "frame_duration_time_weighted_mean_s": squared_duration_sum / duration_sum,
        "occupied_lane_seconds": occupied_seconds,
        "global_barrier_idle_lane_seconds": unassigned_seconds + finished_barrier_seconds,
        "unassigned_lane_seconds": unassigned_seconds,
        "completed_lane_barrier_seconds": finished_barrier_seconds,
        "total_lane_seconds": model.lanes * duration_sum,
        "loaded_frame_count": loaded_frames,
        "by_action_multiset": list(groups.values()),
        "time_regions": {
            name: {
                "start_s": start, "end_s": end,
                "flow_area_job_seconds": sum(region_areas),
                "flow_area_components": dict(zip(COMPONENTS, region_areas, strict=True)),
                "mean_flow_contribution_s": sum(region_areas) / len(tape) if tape else None,
            }
            for name, start, end, region_areas in (
                ("arrival_window", 0.0, arrival_horizon, window_totals),
                ("drain", arrival_horizon, elapsed, drain_totals),
            )
        },
    }
