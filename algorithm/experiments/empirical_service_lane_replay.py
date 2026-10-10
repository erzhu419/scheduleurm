"""Execute empirical complete-frame service while exposing only mean forecasts.

The service stream belongs to the environment.  Policies receive neither its
random generator nor sampled completion/release times for running lanes.
"""
from collections import defaultdict, deque
from dataclasses import asdict
from math import isfinite
from random import Random

from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


class EmpiricalServiceTape:
    """Pair each action's nth execution across policies, regardless of lane.

    Other action draws cannot consume this action's stream.  An action always
    samples its complete duration/offset vector together from one bank record.
    """
    def __init__(self, samples_by_action, seed):
        self.samples = samples_by_action
        self.streams = {action: Random(f"{seed}:action:{action}") for action in samples_by_action}
        self.counts = defaultdict(int)

    def draw(self, option):
        action = option.action_id
        records = self.samples[action]
        index = self.streams[action].randrange(len(records))
        occurrence = self.counts[action]
        self.counts[action] += 1
        return {**records[index], "sample_index": index, "action_occurrence": occurrence}


class MeanService:
    """Execute the same mean catalogue supplied to both policies."""
    def draw(self, option):
        return {"duration": option.duration, "completion_offsets": option.completion_offsets,
                "sample_index": None, "action_occurrence": None}


def simulate_empirical_service_trajectory(model, tape, arrival_horizon, observation_end,
                                         selector, service, *, policy):
    """Retain FIFO, one-second idle polling and complete loaded reservations.

    Event times use actual service.  A busy lane's forecast is its last nominal
    release time clipped at now, even if it is already overdue.  A released
    lane is observed at now.  No conditional residual estimator is introduced.
    ``duration`` in a decision frame remains the policy's nominal duration;
    ``execution_duration`` records the actual occupied lane interval.
    """
    jobs = [{"coordinate": c, "index": index, "arrival": float(time), "start": None, "completion": None}
            for time, c, index in sorted(tape)]
    coordinates = len(model.workloads)
    queues = [deque() for _ in range(coordinates)]
    actual, predicted, observed_counts = [0.] * model.lanes, [0.] * model.lanes, [0] * coordinates
    cursor, frames, lane_events = 0, [], []
    occupied = scheduled_idle = 0.
    work = dict.fromkeys(("root_option_probes", "candidate_count", "rollout_calls", "rollout_option_probes",
                          "eligible_decisions", "changed_decisions", "maximum_rollout_calls",
                          "age_context_job_reads", "observed_arrival_counter_updates",
                          "observed_arrival_context_coordinate_reads"), 0)
    while cursor < len(jobs) or any(queues):
        now = min(actual)
        if now >= observation_end:
            break
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= now:
            coordinate = jobs[cursor]["coordinate"]
            queues[coordinate].append(cursor)
            observed_counts[coordinate] += 1
            work["observed_arrival_counter_updates"] += 1
            cursor += 1
        lane = actual.index(now)
        queue = tuple(map(len, queues))
        before = tuple(now if actual[i] <= now else max(now, predicted[i]) for i in range(model.lanes))
        actual_before = tuple(actual)
        waiting_ages = tuple(tuple(now - jobs[index]["arrival"] for index in waiting) for waiting in queues)
        work["age_context_job_reads"] += sum(queue)
        counts = tuple(observed_counts)
        work["observed_arrival_context_coordinate_reads"] += coordinates
        decision = selector(model.options, queue, before, now, lane, waiting_ages, counts)
        selected = decision.selection
        if any(check_selection(selected, model.options, queue, 1, 0).values()):
            raise ValueError("policy action or nominal support certificate is invalid")
        option_index, = selected.option_indices
        departures = [0] * coordinates
        sampled = None
        if option_index is None:
            execution_duration, offsets = selected.duration, tuple(() for _ in range(coordinates))
        else:
            option = model.options[option_index]
            sampled = service.draw(option)
            raw_duration = float(sampled["duration"])
            offsets = tuple(tuple(group) for group in sampled["completion_offsets"])
            if (not isfinite(raw_duration) or raw_duration <= 0 or len(offsets) != coordinates
                    or any(len(group) != option.requirements[c] or any(
                        not isfinite(offset) or not 0 <= offset <= raw_duration for offset in group)
                        for c, group in enumerate(offsets))):
                raise ValueError("empirical service record does not complete the admitted native jobs")
            execution_duration = max(1., raw_duration)
            for coordinate, group in enumerate(offsets):
                for offset in sorted(group):
                    job = jobs[queues[coordinate].popleft()]
                    job["start"], job["completion"] = now, now + offset
                    departures[coordinate] += 1
        if tuple(departures) != selected.service or any(
                len(queues[c]) != queue[c] - departures[c] for c in range(coordinates)):
            raise ValueError("empirical execution violates FIFO admission or selected service")
        actual[lane], predicted[lane] = now + execution_duration, now + selected.duration
        exposure = max(0., min(actual[lane], observation_end) - now)
        if option_index is None:
            scheduled_idle += exposure
        else:
            occupied += exposure
        lane_events.append({"lane": lane, "start": now, "option_index": option_index,
                            "local_end": actual[lane], "reserved_end": actual[lane]})
        fields = asdict(decision)
        del fields["selection"]
        for key in ("root_option_probes", "candidate_count", "rollout_calls", "rollout_option_probes"):
            work[key] += fields[key]
        work["eligible_decisions"] += fields["eligible"]
        work["changed_decisions"] += fields["changed"]
        work["maximum_rollout_calls"] = max(work["maximum_rollout_calls"], fields["rollout_calls"])
        frames.append({"start": now, "dispatch_time": now, "decision_delay_s": 0.,
                       "queue": queue, "waiting_ages": waiting_ages, "observed_arrival_counts": counts,
                       "available_before": before, "execution_available_before": actual_before,
                       "lane_ids": (lane,), "lanes_called": 1, "option_indices": selected.option_indices,
                       "service": selected.service, "executed_service": tuple(departures),
                       "duration": selected.duration, "execution_duration": execution_duration,
                       "execution_offsets": offsets, "service_sample": sampled,
                       "score": selected.score, "upper_bound": selected.upper_bound, "gap": selected.gap,
                       "score_exact": selected.score_exact, "canonical_exact": selected.canonical_exact,
                       "dispatched": True, "rollout": fields})
    snapshots = {name: trajectory_snapshot(jobs, cutoff, coordinates)
                 for name, cutoff in (("arrival_horizon", arrival_horizon), ("observation_end", observation_end))}
    waiting_area = sum(min(job["start"] if job["start"] is not None else observation_end, observation_end)
                       - job["arrival"] for job in jobs)
    return {"policy": policy, "dispatch_rule": "single", "clock": "empirical_service_hidden_release_mean_forecast",
            "arrival_horizon": arrival_horizon, "observation_end": observation_end,
            "decision_count": len(frames), "frame_count": len(frames), "total_decision_delay_s": 0.,
            "frames": frames, "jobs": jobs, "snapshots": snapshots, "lane_events": lane_events,
            "rollout_work": work, "occupied_lane_time_s": occupied, "scheduled_idle_lane_time_s": scheduled_idle,
            "barrier_idle_lane_time_s": 0., "waiting_area_s": waiting_area, "last_frame_end": max(actual),
            "invariants": {"resource_feasible": True, "completion_offsets_valid": True,
                           "service_matches_departures": True, "queue_recursion": True,
                           "sampled_future_release_times_hidden_from_policy": True}}
