"""Record existing exact BG scores and validate the empirical physical clock."""
from collections import deque
from dataclasses import dataclass, fields
from fractions import Fraction
import json
from math import ceil

from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundCompetitionContinuousLaneRollout, BackgroundCompetitionDecision,
)
from algorithm.experiments.aligned_hardware_background_competition_continuous import EXTRA_WORK
from algorithm.experiments.empirical_service_lane_replay import EmpiricalServiceTape, MeanService
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


@dataclass(frozen=True)
class ExactRecordedDecision(BackgroundCompetitionDecision):
    exact_root_scores: dict | None = None


class ExactRecordedBG(BackgroundCompetitionContinuousLaneRollout):
    """Attach already-computed rational scores; leave selection and RNG intact."""
    def maxweight(self, *args, **kwargs):
        result = super().maxweight(*args, **kwargs)
        self._trace_incumbent, = result.selection.option_indices
        return result

    def score_background_roots(self, *args, **kwargs):
        result = super().score_background_roots(*args, **kwargs)
        self._trace_scores = result
        return result

    def __call__(self, *args, **kwargs):
        self._trace_scores = None
        result = super().__call__(*args, **kwargs)
        scores = None
        if self._trace_scores is not None and not result.planning_overflow:
            baseline, actual = self._trace_incumbent, result.selection.option_indices[0]
            sums = self._trace_scores
            pairs = dict(baseline_S1_sum=sums.costs[baseline], actual_S1_sum=sums.costs[actual],
                         baseline_S2_sum=sums.squared_costs[baseline], actual_S2_sum=sums.squared_costs[actual])
            scores = {key: (value.numerator, value.denominator) for key, value in pairs.items()}
            scores.update(baseline_root=baseline, actual_root=actual)
        return ExactRecordedDecision(**{field.name: getattr(result, field.name)
            for field in fields(BackgroundCompetitionDecision)}, exact_root_scores=scores)


def exact_scores_match(frame):
    info = frame["rollout"]
    scores = info.get("exact_root_scores")
    if not scores or scores["actual_root"] != frame["option_indices"][0]:
        return False
    a, b, a2, b2 = (Fraction(*scores[key]) for key in
        ("actual_S1_sum", "baseline_S1_sum", "actual_S2_sum", "baseline_S2_sum"))
    n = info["sample_count"]
    expected = dict(baseline_predicted_completion_s=float(b / n), chosen_predicted_completion_s=float(a / n),
                    baseline_predicted_squared_flow_s2=float(b2 / n), chosen_predicted_squared_flow_s2=float(a2 / n))
    return (a <= b and a2 <= b2 and info["changed"] == (a < b)
        and (scores["actual_root"] != scores["baseline_root"]) == info["changed"]
        and all(info[key] == value for key, value in expected.items()))


def check_recorded_bg(result, model, protocol, planner):
    # The frozen check_rollout predicates, with strictness checked using the
    # same computation's rational S1/S2 rather than serialized float averages.
    m, r, cap, samples = len(model.options), model.lanes, protocol["planning_event_cap"], protocol["planning_samples"]
    for frame in result["frames"]:
        info, now, counts = frame["rollout"], frame["start"], tuple(frame["observed_arrival_counts"])
        n, ages = sum(frame["queue"]), frame["waiting_ages"]
        if (tuple(map(len, ages)) != tuple(frame["queue"]) or any(
                any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or info["queue_limit"] != r * planner.batch_width
                or info["eligible"] != (0 < n <= info["queue_limit"])):
            return False
        if (tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(value / now if now else 0. for value in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)):
            return False
        k = info["candidate_count"]
        if (not 0 <= k <= m or info["rollout_option_probes"] != m * info["rollout_calls"]
                or info["planning_events"] != info["tagged_planning_events"] + info["background_planning_events"]
                or info["admitted_job_reads"] != info["flow_evaluations"] + info["background_admissions"]
                or info["tag_flag_reads"] != info["admitted_job_reads"]
                or info["background_seed_initializations"] > info["sample_count"]):
            return False
        rescheduled = info["background_planning_events"] - int(info["background_rollout_overflow"])
        positive_rates = sum(value > 0 for value in info["estimated_arrival_rates"])
        if (info["background_planning_rate_reads"] != len(counts) * info["background_seed_initializations"] + rescheduled
                or info["background_planning_random_draws"] != positive_rates * info["background_seed_initializations"] + rescheduled):
            return False
        calls_bound = samples * k * (n + cap - 1 + r * (ceil(planner.maximum_duration) + 1))
        if (info["rollout_calls"] > calls_bound or info["admitted_job_reads"] > samples * k * (n + cap)
                or info["flow_evaluations"] > k * (samples * n + info["tagged_planning_events"])):
            return False
        if not info["eligible"]:
            if info["root_option_probes"] != m or any(info[key] for key in EXTRA_WORK) or info["changed"] or k:
                return False
            continue
        if info["window_s"] != planner.maximum_duration or not 1 <= info["sample_count"] <= samples:
            return False
        if (info["planning_events"] > info["sample_count"] * (cap + 1)
                or info["background_planning_random_draws"] > info["planning_random_draws"]
                or info["background_planning_rate_reads"] > info["planning_rate_reads"]):
            return False
        if info["planning_overflow"]:
            if (info["changed"] or info["baseline_predicted_completion_s"] is not None
                    or info["chosen_predicted_completion_s"] is not None
                    or info["risk_feasible_candidates"] or info["risk_filtered_candidates"]):
                return False
            if info["background_rollout_overflow"]:
                if info["root_option_probes"] != 2 * m or not k or info["sample_count"] != samples:
                    return False
            elif info["root_option_probes"] != m or info["rollout_calls"] or k:
                return False
            continue
        if (info["background_rollout_overflow"] or info["root_option_probes"] != 2 * m
                or info["sample_count"] != samples or not 1 <= info["risk_feasible_candidates"] <= k
                or info["flow_evaluations"] != k * (samples * n + info["tagged_planning_events"])
                or info["risk_filtered_candidates"] + info["risk_feasible_candidates"] != k
                or info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]
                or info["chosen_predicted_squared_flow_s2"] > info["baseline_predicted_squared_flow_s2"]):
            return False
        if not exact_scores_match(frame):
            return False
    return True


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def check_empirical_execution(result, model, protocol, service):
    """Replay saved clock/FIFO/sample records without scoring or simulation."""
    jobs = [{**job, "start": None, "completion": None} for job in result["jobs"]]
    queues = [deque() for _ in model.workloads]
    actual, predicted, counts = [0.] * model.lanes, [0.] * model.lanes, [0] * len(queues)
    cursor, events = 0, []
    draws = (EmpiricalServiceTape(service.samples, result["service_seed"])
             if result["condition"] == "empirical" else MeanService())
    for frame in result["frames"]:
        now, lane = min(actual), actual.index(min(actual))
        if now != frame["start"] or now >= protocol["observation_end"] or frame["lane_ids"] != [lane] and frame["lane_ids"] != (lane,):
            return False
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= now:
            c = jobs[cursor]["coordinate"]
            queues[c].append(cursor)
            counts[c] += 1
            cursor += 1
        ages = tuple(tuple(now - jobs[index]["arrival"] for index in queue) for queue in queues)
        forecast = tuple(now if actual[i] <= now else max(now, predicted[i]) for i in range(model.lanes))
        if not all(_same(frame[key], value) for key, value in (
                ("queue", tuple(map(len, queues))), ("waiting_ages", ages), ("observed_arrival_counts", counts),
                ("available_before", forecast), ("execution_available_before", actual))):
            return False
        index, = frame["option_indices"]
        if index is None:
            sample, duration, nominal, offsets, admitted = None, 1., 1., tuple(() for _ in queues), (0,) * len(queues)
        else:
            option = model.options[index]
            sample = draws.draw(option)
            duration, nominal = max(1., sample["duration"]), max(1., option.duration)
            offsets, admitted = sample["completion_offsets"], option.requirements
            if any(need > len(queue) for need, queue in zip(admitted, queues, strict=True)):
                return False
            for c, group in enumerate(offsets):
                for offset in sorted(group):
                    job = jobs[queues[c].popleft()]
                    job["start"], job["completion"] = now, now + offset
        if not all(_same(frame[key], value) for key, value in (
                ("service_sample", sample), ("execution_duration", duration), ("duration", nominal),
                ("execution_offsets", offsets), ("service", admitted), ("executed_service", admitted))):
            return False
        actual[lane], predicted[lane] = now + duration, now + nominal
        events.append(dict(lane=lane, start=now, option_index=index, local_end=actual[lane], reserved_end=actual[lane]))
    if ((cursor < len(jobs) or any(queues)) and min(actual) < protocol["observation_end"]
            or not _same(jobs, result["jobs"]) or not _same(events, result["lane_events"])):
        return False
    snapshots = {name: trajectory_snapshot(jobs, protocol[key], len(queues)) for name, key in
        (("arrival_horizon", "environment_arrival_end"), ("observation_end", "observation_end"))}
    return _same(snapshots, result["snapshots"])
