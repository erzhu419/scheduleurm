"""Describe saved BG/MW trajectories near a fixed cutoff, without policy calls.

Future completions below are retrospective measurements, never observations
supplied to a policy. Birth-cohort endpoints and completed-at-cutoff are inclusive.
"""
from collections import Counter, defaultdict, deque
from math import fsum, isclose


STATES = ("completed", "queued", "inflight")


def _identity(job):
    return job["coordinate"], job["index"]


def _state(job, time):
    if job["arrival"] > time:
        return "not_arrived"
    if job["completion"] is not None and job["completion"] <= time:
        return "completed"
    return "queued" if job["start"] is None or job["start"] > time else "inflight"


def _interval(start, end, lower, upper):
    return max(0., min(upper, end if end is not None else upper) - max(lower, start))


def _area(jobs, lower, upper):
    queued = fsum(_interval(job["arrival"], job["start"], lower, upper) for job in jobs)
    inflight = fsum(_interval(job["start"], job["completion"], lower, upper)
                   for job in jobs if job["start"] is not None)
    return dict(queued=queued, inflight=inflight, unfinished=queued + inflight)


def _snapshot(jobs, time):
    arrived = [job for job in jobs if job["arrival"] <= time]
    counts = Counter(_state(job, time) for job in arrived)
    unfinished = [job for job in arrived if _state(job, time) != "completed"]
    known = [job for job in unfinished if job["completion"] is not None and job["start"] is not None]
    return dict(arrived=len(arrived), **{key: counts[key] for key in STATES},
        unfinished=counts["queued"] + counts["inflight"],
        unfinished_age_s_sum=fsum(time - job["arrival"] for job in unfinished),
        queued_age_s_sum=fsum(time - job["arrival"] for job in unfinished if _state(job, time) == "queued"),
        known_remaining_sojourn_s=fsum(job["completion"] - time for job in known),
        known_remaining_wait_s=fsum(max(0., job["start"] - time) for job in known),
        known_remaining_execution_s=fsum(job["completion"] - max(time, job["start"]) for job in known),
        remaining_unknown_jobs=len(unfinished) - len(known))


def _grouped(jobs, workloads, function, *args):
    return dict(total=function(jobs, *args), per_workload=[
        dict(coordinate=c, workload=name,
             **function([job for job in jobs if job["coordinate"] == c], *args))
        for c, name in enumerate(workloads)])


def _linked_jobs(result, model):
    """Join admitted jobs to their saved FIFO frames, including simultaneous lanes."""
    jobs = {_identity(job): job for job in result["jobs"]}
    if len(jobs) != len(result["jobs"]):
        raise ValueError("saved jobs contain duplicate identities")
    pending = defaultdict(deque)
    for job in sorted(result["jobs"], key=lambda job: (job["arrival"], *_identity(job))):
        if job["start"] is not None:
            pending[job["coordinate"]].append(job)
    linked, frames = {}, []
    for index, frame in enumerate(result["frames"]):
        option_index, = frame["option_indices"]
        if option_index is None:
            continue
        option = model["options"][option_index]
        start = frame["start"]
        rollout = frame.get("rollout", {})
        roots = rollout.get("exact_root_scores")
        decision = {key: rollout.get(key) for key in ("changed", "eligible", "planning_overflow")}
        if roots:
            for key in ("baseline_root", "actual_root"):
                decision[key] = roots[key]
                decision[key + "_action_id"] = (model["options"][roots[key]]["action_id"]
                                                if roots[key] is not None else None)
        row = dict(frame_index=index, option_index=option_index, action_id=option["action_id"],
            loaded=option["loaded"], requirements=option["requirements"], lane=frame["lane_ids"][0],
            dispatch=start, actual_release=start + frame["execution_duration"],
            nominal_release=start + frame["duration"],
            actual_duration=frame["execution_duration"], nominal_duration=frame["duration"],
            changed=bool(rollout.get("changed", False)), decision=decision, jobs=[])
        for coordinate, offsets in enumerate(frame["execution_offsets"]):
            for offset in sorted(offsets):
                if not pending[coordinate]:
                    raise ValueError("saved frame has no matching admitted FIFO job")
                job = pending[coordinate].popleft()
                if job["start"] != start or job["completion"] != start + offset:
                    raise ValueError("saved FIFO job dispatch/completion does not match frame")
                key = _identity(job)
                row["jobs"].append(key)
                linked[key] = dict(frame_index=index, option_index=option_index,
                    action_id=row["action_id"], loaded=row["loaded"], lane=row["lane"],
                    dispatch=start, actual_release=row["actual_release"],
                    nominal_release=row["nominal_release"], actual_job_offset=offset, decision=decision)
        frames.append(row)
    if any(pending.values()):
        raise ValueError("saved admitted jobs have no matching frame")
    return jobs, linked, frames


def _action_mix(frames, cohort_ids, lower, upper):
    actions = {}
    for frame in frames:
        if not lower <= frame["dispatch"] < upper:
            continue
        action = actions.setdefault(frame["action_id"], dict(action_id=frame["action_id"],
            loaded=frame["loaded"], requirements=frame["requirements"], dispatches=0,
            changed_dispatches=0, admitted_jobs=0, admitted_cohort_jobs=0,
            actual_duration_s=0., nominal_duration_s=0.))
        action["dispatches"] += 1
        action["changed_dispatches"] += frame["changed"]
        action["admitted_jobs"] += len(frame["jobs"])
        action["admitted_cohort_jobs"] += sum(key in cohort_ids for key in frame["jobs"])
        action["actual_duration_s"] += frame["actual_duration"]
        action["nominal_duration_s"] += frame["nominal_duration"]
    return dict(window=[lower, upper], right_endpoint_included=False,
                actions=[actions[key] for key in sorted(actions)])


def _arm(result, model, cohort, cutoff):
    all_jobs, links, frames = _linked_jobs(result, model)
    jobs = [job for job in result["jobs"] if cohort[0] <= job["arrival"] <= cohort[1]]
    ids = {_identity(job) for job in jobs}
    workloads = model["workloads"]
    active = []
    for frame in frames:
        if frame["dispatch"] <= cutoff < frame["actual_release"]:
            cohort_jobs = [all_jobs[key] for key in frame["jobs"] if key in ids]
            active.append(dict(**frame, elapsed_s=cutoff - frame["dispatch"],
                remaining_lane_s=frame["actual_release"] - cutoff,
                nominal_remaining_lane_s=max(0., frame["nominal_release"] - cutoff),
                nominal_overdue_s=max(0., cutoff - frame["nominal_release"]),
                cohort_snapshot=_grouped(cohort_jobs, workloads, _snapshot, cutoff)))
    near_lower, near_upper = max(cohort[0], cutoff - 50.), min(result["observation_end"], cutoff + 50.)
    times = [time for time in (cutoff - 50., cutoff - 25., cutoff, cutoff + 25., cutoff + 50.)
             if cohort[0] <= time <= result["observation_end"]]
    return dict(cohort_jobs=len(jobs), cutoff=_grouped(jobs, workloads, _snapshot, cutoff),
        middle_area_job_s=_grouped(jobs, workloads, _area, cohort[0], cutoff),
        full_area_job_s=_grouped(jobs, workloads, _area, cohort[0], result["observation_end"]),
        neighborhood_area_job_s=_grouped(jobs, workloads, _area, near_lower, near_upper),
        neighborhood_window=[near_lower, near_upper],
        snapshots=[dict(time=time, **_grouped(jobs, workloads, _snapshot, time)) for time in times],
        active_frames=active, action_mix={
            "middle": _action_mix(frames, ids, cohort[0], cutoff),
            "near_cutoff": _action_mix(frames, ids, near_lower, cutoff)}), { _identity(job): job for job in jobs }, links


def _flow_parts(job, end):
    completion = min(end, job["completion"] if job["completion"] is not None else end)
    start = min(end, job["start"] if job["start"] is not None else end)
    flow, wait, execution = completion - job["arrival"], start - job["arrival"], completion - start
    if min(wait, execution) < 0 or not isclose(flow, wait + execution, abs_tol=1e-9):
        raise ValueError("saved job has inconsistent flow/wait/execution decomposition")
    return dict(flow_s=flow, waiting_s=wait, execution_s=execution)


def _decomposition(rows):
    fields = ("flow_s", "waiting_s", "execution_s")
    sums = {key: fsum(row[key] for row in rows) for key in fields}
    return dict(jobs=len(rows), delta_sums=sums,
        delta_means={key: sums[key] / len(rows) if rows else None for key in fields},
        signs={key: dict(improved=sum(row[key] < 0 for row in rows),
                        unchanged=sum(row[key] == 0 for row in rows),
                        worsened=sum(row[key] > 0 for row in rows)) for key in fields},
        max_decomposition_residual_s=max((abs(row["flow_s"] - row["waiting_s"] - row["execution_s"])
                                          for row in rows), default=0.))


def _transition(jobs, other, time):
    counts = {baseline: dict.fromkeys(STATES, 0) for baseline in STATES}
    not_arrived = 0
    for job in jobs:
        baseline, candidate = _state(job, time), _state(other[_identity(job)], time)
        if baseline == "not_arrived":
            not_arrived += 1
        else:
            counts[baseline][candidate] += 1
    return dict(mw_rows_bg_columns=counts, not_arrived=not_arrived)


def analyse_pair(mw_result, bg_result, model_snapshot, cohort=(300, 900), cutoff=900):
    """Return JSON-compatible retrospective statistics for one saved policy pair.

    ``arms`` contains total/per-workload snapshots, job-seconds, active service
    frames and action mix. ``paired`` retains all asymmetric unfinished jobs,
    a complete 3x3 state transition table and per-job time decomposition totals.
    An inconsistent job/frame link or decomposition raises ValueError; such
    output must not be used as diagnostic evidence.
    """
    if mw_result["observation_end"] != bg_result["observation_end"]:
        raise ValueError("paired saved observations must have the same endpoint")
    arms, jobs, links = {}, {}, {}
    for label, result in (("maxweight", mw_result), ("BG", bg_result)):
        arms[label], jobs[label], links[label] = _arm(result, model_snapshot, cohort, cutoff)
    if {key: job["arrival"] for key, job in jobs["maxweight"].items()} != {
            key: job["arrival"] for key, job in jobs["BG"].items()}:
        raise ValueError("paired cohort job identities and arrivals differ")
    workloads, end = model_snapshot["workloads"], mw_result["observation_end"]
    rows, discordant = [], {"bg_only_unfinished": [], "mw_only_unfinished": []}
    unfinished = {label: {key for key, job in group.items()
        if _state(job, cutoff) in ("queued", "inflight")} for label, group in jobs.items()}
    for key in sorted(jobs["maxweight"]):
        values = {label: _flow_parts(group[key], end) for label, group in jobs.items()}
        delta = {field: values["BG"][field] - values["maxweight"][field] for field in values["BG"]}
        if not isclose(delta["flow_s"], delta["waiting_s"] + delta["execution_s"], abs_tol=1e-9):
            raise ValueError("paired job delta flow does not equal delta waiting plus execution")
        rows.append(dict(coordinate=key[0], **delta))
        if (key in unfinished["BG"]) == (key in unfinished["maxweight"]):
            continue
        category = "bg_only_unfinished" if key in unfinished["BG"] else "mw_only_unfinished"
        discordant[category].append(dict(coordinate=key[0], index=key[1], workload=workloads[key[0]],
            arrival=jobs["maxweight"][key]["arrival"], delta=delta,
            arms={label: dict(state=_state(group[key], cutoff), start=group[key]["start"],
                completion=group[key]["completion"], flow_parts=values[label], dispatch=links[label].get(key))
                for label, group in jobs.items()}))
    decomposition = _grouped(rows, workloads, _decomposition)
    area_delta = {key: arms["BG"]["full_area_job_s"]["total"][key]
                      - arms["maxweight"]["full_area_job_s"]["total"][key]
                  for key in ("queued", "inflight", "unfinished")}
    for area, flow in (("queued", "waiting_s"), ("inflight", "execution_s"), ("unfinished", "flow_s")):
        if not isclose(area_delta[area], decomposition["total"]["delta_sums"][flow], abs_tol=1e-7):
            raise ValueError("paired full-cohort area does not match saved job time decomposition")
    return dict(cohort=list(cohort), cutoff=cutoff, observation_end=end, orientation="BG minus maxweight",
        scope="Retrospective saved-trajectory diagnostic; no policy calls, rescoring or new trajectories.",
        arms=arms, paired=dict(
            cutoff_delta={key: arms["BG"]["cutoff"]["total"][key] - arms["maxweight"]["cutoff"]["total"][key]
                          for key in ("completed", "queued", "inflight", "unfinished")},
            state_transitions=_grouped(list(jobs["maxweight"].values()), workloads,
                                       _transition, jobs["BG"], cutoff),
            flow_decomposition=decomposition, full_area_delta_job_s=area_delta,
            shared_unfinished_jobs=len(unfinished["BG"] & unfinished["maxweight"]), **discordant))
