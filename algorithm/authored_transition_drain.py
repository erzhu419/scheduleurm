"""Drain an authored visible queue after one fixed first-frame action.

This helper has no arrival process.  Jobs are already visible at time zero and
may carry nonpositive authored arrival times to represent FIFO age.  The first
selection is supplied by the caller; every later frame uses one common
continuation selector on released queue counts only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import mean
from typing import Any, Callable, Sequence

from algorithm.budgeted_frame_candidates import select_budgeted_frame


@dataclass(frozen=True)
class FixedFrameAction:
    option_indices: tuple[int | None, ...]
    service: tuple[int, ...]
    duration: float
    score: float
    attempts: int = 0
    bound_option_probes: int = 0
    bound_threshold_evaluations: int = 0
    local_score_evaluations: int = 0
    feasible_candidate_count: int = 0


def fixed_frame_action(model: Any, queues: Sequence[int], option_indices) -> FixedFrameAction:
    """Materialize one feasible raw lane tuple for an authored execution check."""
    indices = tuple(option_indices)
    if len(queues) != len(model.workloads) or any(
        int(value) != value or value < 0 for value in queues
    ):
        raise ValueError("queues must match the model's nonnegative integer coordinates")
    queue = tuple(int(value) for value in queues)
    if len(indices) != model.lanes:
        raise ValueError("fixed action lane count differs from the model")
    if any(index is not None and not 0 <= index < len(model.options) for index in indices):
        raise ValueError("fixed action contains an unknown option index")
    service = tuple(
        sum(
            model.options[index].requirements[coordinate]
            for index in indices
            if index is not None
        )
        for coordinate in range(len(queue))
    )
    if any(count > available for count, available in zip(service, queue, strict=True)):
        raise ValueError("fixed action exceeds the visible queue")
    duration = max((
        1.0,
        *(model.options[index].duration for index in indices if index is not None),
    ))
    score = sum(q * count for q, count in zip(queue, service, strict=True)) / duration
    return FixedFrameAction(indices, service, duration, score)


def authored_arrival_times(
    queues: Sequence[int], profile: str
) -> tuple[tuple[float, ...], ...]:
    """Create one frozen FIFO-age profile for a visible queue."""
    if any(int(value) != value or value < 0 for value in queues):
        raise ValueError("queues must contain nonnegative integer job counts")
    counts = tuple(int(value) for value in queues)
    if profile == "coincident":
        return tuple((0.0,) * count for count in counts)
    if profile == "unit_fifo":
        return tuple(
            tuple(float(index - count + 1) for index in range(count))
            for count in counts
        )
    raise ValueError("profile must be 'coincident' or 'unit_fifo'")


def drain_initial_population(
    model: Any,
    arrival_times: Sequence[Sequence[float]],
    budget: int,
    first_action: Any,
    *,
    continuation: Callable[..., Any] = select_budgeted_frame,
) -> dict:
    """Execute one fixed action, then a common selector until the queue drains."""
    if len(arrival_times) != len(model.workloads):
        raise ValueError("arrival-time coordinates must match the model workloads")
    normalized = []
    for times in arrival_times:
        values = tuple(float(value) for value in times)
        if any(not math.isfinite(value) or value > 0.0 for value in values):
            raise ValueError("authored arrival times must be finite and nonpositive")
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("authored arrival times must be in FIFO order")
        normalized.append(values)
    queues = [
        deque((arrival, index) for index, arrival in enumerate(times))
        for times in normalized
    ]
    job_count = sum(len(queue) for queue in queues)
    if job_count == 0:
        raise ValueError("the authored transition requires a nonempty queue")

    now = 0.0
    frame_index = 0
    initial_age_area = -sum(sum(times) for times in normalized)
    deferred_area = 0.0
    dispatch_area = 0.0
    completion_records = []
    frame_trace = []
    work_fields = (
        "attempts",
        "bound_option_probes",
        "bound_threshold_evaluations",
        "local_score_evaluations",
        "feasible_candidate_count",
        "cache_hits",
        "logical_search_pops",
        "upper_bound_calls",
    )
    work = dict.fromkeys(work_fields, 0)

    while any(queues):
        before = tuple(len(queue) for queue in queues)
        selected = (
            first_action
            if frame_index == 0
            else continuation(model.options, before, model.lanes, budget)
        )
        if len(selected.option_indices) != model.lanes:
            raise ValueError("selected lane count differs from the model")
        duration = float(selected.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("selected duration must be positive and finite")
        if sum(selected.service) <= 0:
            raise ValueError("a nonempty authored queue selected no service")
        for name in work_fields:
            work[name] += getattr(selected, name, 0)

        departures = [0] * len(queues)
        dispatch_this_frame = 0.0
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            for coordinate, offsets in enumerate(option.completion_offsets):
                for offset in offsets:
                    offset = float(offset)
                    if offset < 0.0 or offset > duration:
                        raise ValueError("completion offset lies outside its frame")
                    if not queues[coordinate]:
                        raise ValueError("selected action exceeds the visible FIFO queue")
                    arrival, job_index = queues[coordinate].popleft()
                    completion = now + offset
                    flow = completion - arrival
                    initial_age = -arrival
                    completion_records.append({
                        "coordinate": coordinate,
                        "job_index": job_index,
                        "arrival_s": arrival,
                        "completion_s": completion,
                        "flow_s": flow,
                        "initial_age_s": initial_age,
                        "deferred_visible_wait_s": now,
                        "dispatch_to_completion_s": offset,
                        "frame_index": frame_index,
                        "option_index": option_index,
                        "completion_offset_s": offset,
                    })
                    dispatch_this_frame += offset
                    departures[coordinate] += 1
        if tuple(departures) != tuple(selected.service):
            raise ValueError("selected service differs from executed departures")
        deferred_this_frame = sum(
            (count - served) * duration
            for count, served in zip(before, departures, strict=True)
        )
        deferred_area += deferred_this_frame
        dispatch_area += dispatch_this_frame
        after = tuple(len(queue) for queue in queues)
        frame_trace.append({
            "frame_index": frame_index,
            "start_s": now,
            "duration_s": duration,
            "queue_before": before,
            "departures": tuple(departures),
            "queue_after": after,
            "option_indices": tuple(selected.option_indices),
            "score": float(selected.score),
            "attempts": int(selected.attempts),
            "deferred_visible_area_job_s": deferred_this_frame,
            "dispatch_to_completion_area_job_s": dispatch_this_frame,
        })
        now += duration
        frame_index += 1
        if frame_index > job_count:
            raise ValueError("authored population failed to drain within one service per frame")

    flows = [row["flow_s"] for row in completion_records]
    ordered_flows = sorted(flows)
    direct_area = sum(flows)
    decomposed_area = initial_age_area + deferred_area + dispatch_area
    if len(flows) != job_count:
        raise ValueError("authored drain did not complete the initial population")
    if not math.isclose(direct_area, decomposed_area, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("authored flow-area identity failed")
    return {
        "job_count": job_count,
        "completed_jobs": len(flows),
        "frame_count": frame_index,
        "continuation_selector_calls": frame_index - 1,
        "elapsed_frame_barriers_s": now,
        "makespan_s": max(row["completion_s"] for row in completion_records),
        "mean_flow_s": mean(flows),
        "p90_flow_s": ordered_flows[math.ceil(0.9 * job_count) - 1],
        "maximum_flow_s": ordered_flows[-1],
        "flow_area_job_s": direct_area,
        "flow_area_components_job_s": {
            "initial_age": initial_age_area,
            "deferred_visible": deferred_area,
            "dispatch_to_completion": dispatch_area,
        },
        "flow_area_identity_error_job_s": direct_area - decomposed_area,
        "selection_work": work,
        "frame_trace": frame_trace,
        "completion_records": completion_records,
    }
