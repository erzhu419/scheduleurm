"""Compare lane dispatch rules on one causal, service-only FIFO clock."""
from __future__ import annotations

from collections import deque
from math import isfinite
from time import perf_counter

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, check_selection,
)
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


def simulate_lane_refill_trajectory(
    model, tape, arrival_horizon, observation_end, budget, selector, *, policy,
    dispatch_rule,
):
    """Dispatch only on lane-availability events; arrivals never trigger a call.

    Busy lanes retain their admitted jobs until their complete local option
    ends. Earlier job completion offsets do not release a loaded lane. Idle
    lanes poll after the original one-second interval. Selector runtime is
    auxiliary accounting and never advances this experiment's service clock.
    """
    if dispatch_rule not in ("synchronous", "refill", "single"):
        raise ValueError("dispatch_rule must be synchronous, refill, or single")
    if not (0 <= arrival_horizon <= observation_end and isfinite(observation_end)):
        raise ValueError("require finite 0 <= arrival_horizon <= observation_end")
    coordinates = len(model.workloads)
    events = tuple(sorted(tape))
    if any(not isfinite(time) or not 0 <= time <= arrival_horizon or
           not 0 <= coordinate < coordinates for time, coordinate, _ in events):
        raise ValueError("arrival tape is outside registered horizon or coordinates")
    if len({(coordinate, index) for _, coordinate, index in events}) != len(events):
        raise ValueError("arrival identities must be unique within coordinates")
    jobs = [{"coordinate": coordinate, "index": index, "arrival": float(time),
             "start": None, "completion": None} for time, coordinate, index in events]
    queues = [deque() for _ in range(coordinates)]
    available = [0.0] * model.lanes
    frames, originals, lane_events = [], [], []
    cursor = 0
    selector_wall_s = occupied = barrier_idle = scheduled_idle = 0.0
    invariants = dict.fromkeys(("resource_feasible", "completion_offsets_valid",
                                "service_matches_departures", "queue_recursion",
                                "lane_nonoverlap"), True)

    def release(until):
        nonlocal cursor
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= until:
            queues[jobs[cursor]["coordinate"]].append(cursor)
            cursor += 1

    def exposure(start, end):
        return max(0.0, min(end, observation_end) - start)

    # Once every job is admitted, record its scheduled completion without
    # introducing idle calls solely to advance an otherwise drained clock.
    while cursor < len(jobs) or any(queues):
        now = min(available)
        if now >= observation_end:
            break
        release(now)
        free = tuple(lane for lane, end in enumerate(available) if end == now)
        lanes = free[:1] if dispatch_rule == "single" else free
        queue = tuple(len(waiting) for waiting in queues)
        started = perf_counter()
        result = selector(model.options, queue, len(lanes), budget)
        elapsed = perf_counter() - started
        selected = getattr(result, "selection", result)
        checks = check_selection(selected, model.options, queue, len(lanes),
                                 selected.lp_node_budget)
        if any(checks[key] for key in (
            "invalid_counts", "index_count_mismatch", "service_count_mismatch",
            "infeasible_service", "duration_count_mismatch",
        )):
            raise ValueError(f"infeasible selected lane batch at {now}: {checks}")
        if not isfinite(selected.duration) or selected.duration <= 0:
            raise ValueError("selected frame must advance time")
        class_offsets = [[] for _ in range(coordinates)]
        pending_events = []
        for lane, option_index in zip(lanes, selected.option_indices, strict=True):
            local_duration = 1.0
            if option_index is not None:
                option = model.options[option_index]
                if len(option.completion_offsets) != coordinates or any(
                    len(offsets) != option.requirements[c] or any(
                        not isfinite(offset) or not 0 <= offset <= option.duration
                        for offset in offsets)
                    for c, offsets in enumerate(option.completion_offsets)
                ):
                    raise ValueError(f"invalid completion offsets for {option.action_id}")
                local_duration = max(1.0, float(option.duration))
                for coordinate, offsets in enumerate(option.completion_offsets):
                    class_offsets[coordinate].extend(offsets)
            local_end = now + local_duration
            reserved_end = (now + selected.duration if dispatch_rule == "synchronous"
                            else local_end)
            pending_events.append({"lane": lane, "start": now,
                                   "option_index": option_index,
                                   "local_end": local_end, "reserved_end": reserved_end})

        departures = [0] * coordinates
        # All jobs in a class have the same canonical size. Bind older FIFO
        # admissions to earlier slots across this call's selected local options.
        for coordinate, offsets in enumerate(class_offsets):
            for offset in sorted(offsets):
                job = jobs[queues[coordinate].popleft()]
                job["start"], job["completion"] = now, now + offset
                departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise ValueError("selected service differs from executed departures")
        after = tuple(len(waiting) for waiting in queues)
        if any(q1 != q0 - done for q0, done, q1 in
               zip(queue, departures, after, strict=True)):
            raise ValueError("admission queue recursion failed")

        for event in pending_events:
            lane = event["lane"]
            if available[lane] > now:
                raise ValueError("lane reused before its reservation ends")
            available[lane] = event["reserved_end"]
            local_time = exposure(now, event["local_end"])
            if event["option_index"] is None:
                scheduled_idle += local_time
            else:
                occupied += local_time
            barrier_idle += exposure(event["local_end"], event["reserved_end"])
            lane_events.append(event)
        selector_wall_s += elapsed
        originals.append(result)
        frames.append({
            "start": now, "dispatch_time": now, "decision_delay_s": 0.0,
            "queue": queue, "lane_ids": lanes, "lanes_called": len(lanes),
            "option_indices": selected.option_indices, "service": selected.service,
            "executed_service": tuple(departures), "duration": selected.duration,
            "score": selected.score, "score_exact": selected.score_exact,
            "canonical_exact": selected.canonical_exact, "gap": selected.gap,
            "search_work": {counter: getattr(selected, counter) for counter in COUNTERS},
            "selector_wall_s": elapsed, "dispatched": True,
        })

    snapshots = {label: trajectory_snapshot(jobs, cutoff, coordinates)
                 for label, cutoff in (("arrival_horizon", arrival_horizon),
                                       ("observation_end", observation_end))}
    waiting_area = sum(min(job["start"] if job["start"] is not None else observation_end,
                           observation_end) - job["arrival"] for job in jobs)
    trajectory = {
        "policy": policy, "dispatch_rule": dispatch_rule, "budget": budget,
        "clock": "physical_service_only_no_compute_delay_one_second_idle_polling",
        "arrival_horizon": arrival_horizon, "observation_end": observation_end,
        "frame_count": len(frames), "decision_count": len(frames),
        "executed_frame_count": len(frames),
        "total_decision_delay_s": 0.0,
        "last_frame_end": max(available, default=0.0),
        "last_decision_end": frames[-1]["dispatch_time"] if frames else None,
        "frames": frames, "jobs": jobs, "snapshots": snapshots,
        "lane_events": lane_events, "invariants": invariants,
        "occupied_lane_time_s": occupied, "barrier_idle_lane_time_s": barrier_idle,
        "scheduled_idle_lane_time_s": scheduled_idle, "waiting_area_s": waiting_area,
        "selector_wall_s": selector_wall_s, "total_selector_wall_s": selector_wall_s,
    }
    return trajectory, originals
