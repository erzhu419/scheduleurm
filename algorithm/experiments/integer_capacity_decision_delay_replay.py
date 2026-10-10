"""Causal FIFO replay charging each measured cold selection to its own clock.

Only the selector is timed. Reference solves run in a separate offline audit,
after every measured policy block, and never contribute simulated delay.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from math import isfinite
from time import perf_counter_ns

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, _metrics, _record, check_selection, reference_key,
)
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


def simulate_delayed_trajectory(model, tape, arrival_horizon, observation_end, budget,
                                selector, policy="candidate", clock=perf_counter_ns):
    """Return a replay and its unencoded selections for a later reference audit.

    The queue snapshot precedes selection. Arrivals during the measured solve
    join the waiting queues but cannot enter that decision's frame. A decision
    finishing at or after T is paid in full and never dispatches any jobs.
    """
    if not (0 <= arrival_horizon <= observation_end and isfinite(observation_end)):
        raise ValueError("require finite 0 <= arrival_horizon <= observation_end")
    coordinates = len(model.workloads)
    events = tuple(sorted(tape))
    if any(not isfinite(time) or not 0 <= time <= arrival_horizon or
           not 0 <= coordinate < coordinates for time, coordinate, _ in events):
        raise ValueError("arrival tape is outside the registered horizon or coordinates")
    if len({(coordinate, index) for _, coordinate, index in events}) != len(events):
        raise ValueError("arrival identities must be unique within coordinates")
    jobs = [{"coordinate": coordinate, "index": index, "arrival": float(time),
             "start": None, "completion": None} for time, coordinate, index in events]
    queues = [deque() for _ in range(coordinates)]
    invariants = dict.fromkeys(("resource_feasible", "completion_offsets_valid",
                                "service_matches_departures", "queue_recursion"), True)
    frames, selections = [], []
    cursor = 0
    now = 0.0

    def release(until):
        nonlocal cursor
        counts = [0] * coordinates
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= until:
            coordinate = jobs[cursor]["coordinate"]
            queues[coordinate].append(cursor)
            counts[coordinate] += 1
            cursor += 1
        return tuple(counts)

    release(now)
    while now < observation_end and (cursor < len(jobs) or any(queues)):
        queue = tuple(len(waiting) for waiting in queues)
        started = clock()
        result = selector(model.options, queue, model.lanes, budget)
        ended = clock()
        delay = (ended - started) / 1_000_000_000
        if not isfinite(delay) or delay < 0:
            raise ValueError("selector clock must return finite nondecreasing nanoseconds")
        selected = getattr(result, "selection", result)
        checks = check_selection(selected, model.options, queue, model.lanes, budget)
        if any(checks[key] for key in ("invalid_counts", "index_count_mismatch",
                                       "service_count_mismatch", "infeasible_service",
                                       "duration_count_mismatch")):
            raise ValueError(f"infeasible selected frame at {now}: {checks}")
        if not isfinite(selected.duration) or selected.duration <= 0:
            raise ValueError("selected frame must advance time")
        class_offsets = [[] for _ in range(coordinates)]
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            if len(option.completion_offsets) != coordinates or any(
                len(offsets) != option.requirements[c] or any(
                    not isfinite(offset) or not 0 <= offset <= option.duration for offset in offsets)
                for c, offsets in enumerate(option.completion_offsets)
            ):
                raise ValueError(f"invalid completion offsets for {option.action_id}")
            for coordinate, offsets in enumerate(option.completion_offsets):
                class_offsets[coordinate].extend(offsets)
        if tuple(map(len, class_offsets)) != selected.service:
            raise ValueError("selected service differs from completion-slot counts")

        dispatch = now + delay
        arrivals_during_solve = release(min(dispatch, observation_end))
        dispatched = dispatch < observation_end
        departures = [0] * coordinates
        if dispatched:
            # Earlier releases stay ahead of all arrivals during this solve.
            for coordinate, offsets in enumerate(class_offsets):
                for offset in sorted(offsets):
                    job = jobs[queues[coordinate].popleft()]
                    if job["arrival"] > now:
                        raise ValueError("selection used an arrival after its queue snapshot")
                    job["start"], job["completion"] = dispatch, dispatch + offset
                    departures[coordinate] += 1
            if tuple(departures) != selected.service:
                raise ValueError("selected service differs from executed departures")
            end = dispatch + selected.duration
            arrivals_during_frame = release(min(end, observation_end))
        else:
            end = observation_end
            arrivals_during_frame = (0,) * coordinates
        after = tuple(len(waiting) for waiting in queues)
        if any(q1 != q0 - done + during_solve + during_frame
               for q0, done, during_solve, during_frame, q1 in zip(
                   queue, departures, arrivals_during_solve, arrivals_during_frame,
                   after, strict=True)):
            raise ValueError("queue recursion failed")
        frames.append({
            "start": now, "dispatch_time": dispatch, "decision_delay_s": delay,
            "duration": selected.duration, "queue": queue,
            "option_indices": selected.option_indices, "service": selected.service,
            "executed_service": tuple(departures), "dispatched": dispatched,
            "score": selected.score, "score_exact": selected.score_exact,
            "canonical_exact": selected.canonical_exact, "gap": selected.gap,
            "search_work": {counter: getattr(selected, counter) for counter in COUNTERS},
        })
        selections.append(result)
        now = end
    snapshots = {label: trajectory_snapshot(jobs, cutoff, coordinates) for label, cutoff in (
        ("arrival_horizon", arrival_horizon), ("observation_end", observation_end),
    )}
    return {
        "policy": policy, "budget": budget, "arrival_horizon": arrival_horizon,
        "observation_end": observation_end, "frame_count": len(frames),
        "decision_count": len(frames),
        "executed_frame_count": sum(frame["dispatched"] for frame in frames),
        "total_decision_delay_s": sum(frame["decision_delay_s"] for frame in frames),
        "last_frame_end": now,
        "last_decision_end": frames[-1]["dispatch_time"] if frames else None,
        "frames": frames, "jobs": jobs, "snapshots": snapshots,
        "invariants": invariants,
    }, selections


def audit_delayed_trajectory(model, trajectory, selections, reference_cache, audit_limit=4):
    """Attach offline correctness/work accounting, including censored decisions."""
    frames = trajectory["frames"]
    if len(frames) != len(selections):
        raise ValueError("each paid decision needs its original selection for audit")
    budget = trajectory["budget"]
    before = reference_cache.summary()
    metrics = _metrics()
    metrics["capacity_propagation"] = {}
    complete = incomplete = 0
    first_uncertified = first_incomplete = None
    witnesses = []
    for number, (frame, result) in enumerate(zip(frames, selections, strict=True)):
        selected = getattr(result, "selection", result)
        queue = tuple(frame["queue"])
        reference = (reference_cache.observe(queue, selected)
                     if trajectory["policy"] == "full_reference"
                     else reference_cache.lookup(queue))
        expected = reference_key(reference, queue, len(model.options))
        complete += expected is not None
        incomplete += expected is None
        if expected is None and first_incomplete is None:
            first_incomplete = {"frame": number, "start": frame["start"],
                                "queue": queue, "selection": asdict(reference)}
        preprocessing = asdict(result.preprocessing) if hasattr(result, "preprocessing") else None
        _record(metrics, selected, model, queue, model.lanes, budget, expected,
                preprocessing, reference, frame["decision_delay_s"])
        if hasattr(result, "capacity_propagation"):
            for key, value in asdict(result.capacity_propagation).items():
                metrics["capacity_propagation"][key] = metrics["capacity_propagation"].get(key, 0) + value
        if not selected.canonical_exact and first_uncertified is None:
            first_uncertified = {"frame": number, "start": frame["start"],
                                 "queue": queue, "selection": asdict(selected)}
        checks = check_selection(selected, model.options, queue, model.lanes,
                                 budget, expected, reference)
        if len(witnesses) < audit_limit and (any(checks.values()) or not selected.canonical_exact):
            witnesses.append({**frame, "checks": {key: value for key, value in checks.items() if value},
                              "reference_complete": expected is not None,
                              "reference_indices": reference.option_indices})
    after = reference_cache.summary()
    audit = {key: after[key] - before[key] for key in (
        "lookup_calls", "cache_hits", "observed_policy_calls", "actual_calls",
        "operational_elapsed_s", "certificate_failures", "calls", "failures",
    )}
    audit.update({
        "search_work": {key: after["search_work"][key] - before["search_work"][key]
                        for key in COUNTERS},
        "cache_size": after["cache_size"], "complete_frame_count": complete,
        "incomplete_frame_count": incomplete, "incomplete_count": incomplete,
        "first_incomplete": first_incomplete,
    })
    trajectory.update({
        "selector_metrics": metrics, "reference_audit": audit,
        "first_uncertified_frame": first_uncertified, "first_witnesses": witnesses,
    })
    return trajectory
