"""Offline same-clock continuation from an actual pre-action checkpoint."""
from collections import deque
from dataclasses import asdict

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


def resume_single_action_trajectory(model, arrival_horizon, observation_end, selector, checkpoint, *, policy):
    """Dispatch on the original service clock with seven-argument context.

    Waiting ages exclude admitted jobs; cumulative observed arrival counts
    include them. Each count changes once when the corresponding birth is
    released. Earlier calls at the same timestamp leave the counts intact.
    Arrivals never trigger dispatch, idle lanes poll after one second, and
    loaded actions reserve their complete duration.
    """
    jobs = [dict(job) for job in checkpoint["jobs"]]
    coordinates = len(model.workloads)
    queues = [deque(group) for group in checkpoint["queues"]]
    available = list(checkpoint["available"])
    observed_counts = list(checkpoint["observed_counts"])
    cursor = checkpoint["cursor"]
    frames, lane_events = [], []
    occupied = scheduled_idle = 0.0
    work = dict.fromkeys(("root_option_probes", "candidate_count", "rollout_calls", "rollout_option_probes",
                          "eligible_decisions", "changed_decisions", "maximum_rollout_calls",
                          "age_context_job_reads", "observed_arrival_counter_updates",
                          "observed_arrival_context_coordinate_reads"), 0)
    while cursor < len(jobs) or any(queues):
        now = min(available)
        if now >= observation_end:
            break
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= now:
            coordinate = jobs[cursor]["coordinate"]
            queues[coordinate].append(cursor)
            observed_counts[coordinate] += 1
            work["observed_arrival_counter_updates"] += 1
            cursor += 1
        lane = available.index(now)
        queue = tuple(len(waiting) for waiting in queues)
        before = tuple(available)
        waiting_ages = tuple(tuple(now - jobs[index]["arrival"] for index in waiting)
                             for waiting in queues)
        work["age_context_job_reads"] += sum(queue)
        observed_arrival_counts = tuple(observed_counts)
        work["observed_arrival_context_coordinate_reads"] += coordinates
        decision = selector(model.options, queue, before, now, lane, waiting_ages, observed_arrival_counts)
        selected = decision.selection
        if any(check_selection(selected, model.options, queue, 1, 0).values()):
            raise ValueError("observed-rate policy returned an infeasible action or an invalid support certificate")
        option_index, = selected.option_indices
        departures = [0] * coordinates
        if option_index is not None:
            option = model.options[option_index]
            if any(len(offsets) != option.requirements[c] or any(not 0 <= offset <= option.duration for offset in offsets)
                   for c, offsets in enumerate(option.completion_offsets)):
                raise ValueError("complete-job offsets differ from admitted service")
            for c, offsets in enumerate(option.completion_offsets):
                for offset in sorted(offsets):
                    job = jobs[queues[c].popleft()]
                    job["start"], job["completion"] = now, now + offset
                    departures[c] += 1
        if tuple(departures) != selected.service:
            raise ValueError("executed departures differ from the selected service")
        if any(len(queues[c]) != queue[c] - departures[c] for c in range(coordinates)):
            raise ValueError("FIFO admission queue recursion failed")
        available[lane] = now + selected.duration
        exposure = max(0.0, min(available[lane], observation_end) - now)
        if option_index is None:
            scheduled_idle += exposure
        else:
            occupied += exposure
        lane_events.append({"lane": lane, "start": now, "option_index": option_index,
                            "local_end": available[lane], "reserved_end": available[lane]})
        fields = asdict(decision)
        del fields["selection"]
        for key in ("root_option_probes", "candidate_count", "rollout_calls", "rollout_option_probes"):
            work[key] += fields[key]
        work["eligible_decisions"] += fields["eligible"]
        work["changed_decisions"] += fields["changed"]
        work["maximum_rollout_calls"] = max(work["maximum_rollout_calls"], fields["rollout_calls"])
        frames.append({"start": now, "dispatch_time": now, "decision_delay_s": 0.0,
                       "queue": queue, "waiting_ages": waiting_ages,
                       "observed_arrival_counts": observed_arrival_counts,
                       "available_before": before, "lane_ids": (lane,), "lanes_called": 1,
                       "option_indices": selected.option_indices, "service": selected.service,
                       "executed_service": tuple(departures), "duration": selected.duration,
                       "score": selected.score, "upper_bound": selected.upper_bound, "gap": selected.gap,
                       "score_exact": selected.score_exact, "canonical_exact": selected.canonical_exact,
                       "dispatched": True, "rollout": fields})
    snapshots = {name: trajectory_snapshot(jobs, cutoff, coordinates)
                 for name, cutoff in (("arrival_horizon", arrival_horizon), ("observation_end", observation_end))}
    waiting_area = sum(min(job["start"] if job["start"] is not None else observation_end, observation_end)
                       - job["arrival"] for job in jobs)
    return {"policy": policy, "dispatch_rule": "single", "clock": "physical_service_only_no_compute_delay_one_second_idle_polling",
            "arrival_horizon": arrival_horizon, "observation_end": observation_end,
            "decision_count": len(frames), "frame_count": len(frames), "total_decision_delay_s": 0.0, "frames": frames, "jobs": jobs,
            "snapshots": snapshots, "lane_events": lane_events, "rollout_work": work,
            "occupied_lane_time_s": occupied, "scheduled_idle_lane_time_s": scheduled_idle,
            "barrier_idle_lane_time_s": 0.0, "waiting_area_s": waiting_area,
            "last_frame_end": max(available), "invariants": {"resource_feasible": True, "completion_offsets_valid": True,
                                                            "service_matches_departures": True, "queue_recursion": True}}


def prepare_checkpoint(factual, frame_index):
    """Preserve only prefix admissions, including busy committed completions."""
    frame = factual["frames"][frame_index]
    dimension = len(frame["queue"])
    admitted = [sum(previous["service"][c] for previous in factual["frames"][:frame_index])
                for c in range(dimension)]
    jobs = [dict(job) for job in factual["jobs"]]
    by_class = [[i for i, job in enumerate(jobs) if job["coordinate"] == c]
                for c in range(dimension)]
    for c, indices in enumerate(by_class):
        for index in indices[admitted[c]:]:
            jobs[index]["start"] = jobs[index]["completion"] = None
    observed = frame["observed_arrival_counts"]
    queues = [indices[admitted[c]:observed[c]] for c, indices in enumerate(by_class)]
    assert tuple(map(len, queues)) == tuple(frame["queue"])
    assert tuple(tuple(frame["start"] - jobs[i]["arrival"] for i in group) for group in queues) == tuple(tuple(group) for group in frame["waiting_ages"])
    return {"jobs": jobs, "queues": queues, "available": frame["available_before"],
            "observed_counts": observed, "cursor": sum(observed),
            "checkpoint_job_reads": len(jobs) * (dimension + 1),
            "checkpoint_prefix_service_reads": frame_index * dimension}
