"""Restore a cached H state and continue its original complete finite policy.

Only the terminal controller changes. The H population, pre-H cost and external
arrival stream stay fixed; later arrivals compete without contributing cost.
"""
from collections import Counter, deque
from fractions import Fraction
from time import perf_counter, process_time

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, STATUS
from algorithm.experiments.competing_terminal_flow_rollout import _fraction
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


def _simulate_native_terminal(planner, jobs, terminal_state, suffix, *, origin, horizon,
        common_area, waiting_area, inflight_area, progress=None):
    """Run physical terminal dispatches using an already restored native planner."""
    cpu, wall = process_time(), perf_counter()
    jobs = [dict(job) for job in jobs]
    by_id = {tuple(job["id"]): job for job in jobs}
    waiting = [deque() for _ in terminal_state["queue"]]
    waiting_ids = {tuple(identity) for group in terminal_state["waiting_ids"] for identity in group}
    for coordinate, group in enumerate(terminal_state["waiting_ids"]):
        for identity in group:
            job = by_id[tuple(identity)]
            if job["coordinate"] != coordinate:
                raise ValueError("cached waiting identity differs from its FIFO coordinate")
            job["start"] = job["completion"] = None
            waiting[coordinate].append(job)
    if tuple(map(len, waiting)) != tuple(terminal_state["queue"]):
        raise ValueError("cached H FIFO differs from its queue counts")
    if any(job["completion"] is None for job in jobs if tuple(job["id"]) not in waiting_ids):
        raise ValueError("a nonwaiting cached job lacks its committed completion")
    remaining = len(waiting_ids)
    lane_ends = [max(horizon, end) for end in terminal_state["lane_ends"]]
    counts = list(terminal_state["counts"])
    background_cursor = len(suffix.events)
    before_suffix = (len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    logical_work, statuses, dispatch_work = Counter(), Counter(), Counter()
    calls = idle_calls = competing_admissions = 0
    overflow = False
    partial_end = horizon
    first_call_context = first_native_decision = None
    last_progress = wall - 30

    def announce(force=False, proposal_wall_s=None):
        nonlocal last_progress
        now = perf_counter()
        if progress is not None and (force or now - last_progress >= 30):
            progress(dict(phase="native_terminal", remaining_fixed_jobs=remaining,
                calls=calls, idle_calls=idle_calls, competing_admissions=competing_admissions,
                background_events=len(suffix.events), elapsed_wall_s=now - wall,
                elapsed_cpu_s=process_time() - cpu, proposal_wall_s=proposal_wall_s))
            last_progress = now

    announce(force=True)
    while remaining:
        lane = min(range(len(lane_ends)), key=lambda index: (lane_ends[index], index))
        start = lane_ends[lane]
        stream_complete = suffix.release_through(start)
        while background_cursor < len(suffix.events):
            birth, coordinate = suffix.events[background_cursor]
            if birth > start:
                break
            waiting[coordinate].append(dict(id=("post_H_background", background_cursor),
                coordinate=coordinate, birth=birth, source="post_H_background", start=None, completion=None))
            counts[coordinate] += 1
            dispatch_work["observed_arrival_counter_updates"] += 1
            background_cursor += 1
        if not stream_complete:
            overflow = True
            partial_end = suffix.events[-1][0]
            break
        queue = tuple(map(len, waiting))
        ages = tuple(tuple(float(start - job["birth"]) for job in group) for group in waiting)
        context = (planner.options, queue, tuple(float(origin + end) for end in lane_ends),
            float(origin + start), lane, ages, tuple(counts))
        dispatch_work["age_context_job_reads"] += sum(queue)
        dispatch_work["observed_arrival_context_coordinate_reads"] += len(counts)
        dispatch_work["lane_context_reads"] += len(lane_ends)
        announce()
        proposal_begin = perf_counter()
        decision = planner(*context)
        calls += 1
        logical_work.update({name: getattr(decision, name) for name in FEES + TWO_STEP_WORK})
        for work in (decision.inner_current_work, decision.inner_current_fallbacks):
            logical_work.update({"inner_current_" + name: value for name, value in work.items()})
        statuses.update({name: getattr(decision, name) for name in STATUS})
        if calls == 1:
            first_call_context, first_native_decision = context[1:], decision
        chosen, = decision.selection.option_indices
        if chosen is None:
            idle_calls += 1
            lane_ends[lane] = start + 1
        else:
            for coordinate, offsets in enumerate(planner.sorted_completion_offsets[chosen]):
                for offset in offsets:
                    job = waiting[coordinate].popleft()
                    job["start"], job["completion"] = start, start + offset
                    dispatch_work["fifo_identity_admissions"] += 1
                    if job["source"] == "post_H_background":
                        competing_admissions += 1
                    else:
                        remaining -= 1
                        dispatch_work["fixed_population_completion_evaluations"] += 1
            lane_ends[lane] = start + planner.durations[chosen]
        partial_end = start
        announce(proposal_wall_s=perf_counter() - proposal_begin)

    def components(job):
        begin = max(Fraction(0), job["birth"])
        start, completion = job["start"], job["completion"]
        w = max(Fraction(0), min(horizon, start if start is not None else horizon) - begin)
        b = (max(Fraction(0), min(horizon, completion) - max(Fraction(0), start))
            if completion is not None else Fraction(0))
        residual = (max(Fraction(0), completion - horizon) if completion is not None
            else max(Fraction(0), partial_end - horizon))
        return w, b, w + b, residual

    def group_result(group):
        rows = [components(job) for job in group]
        w, b, area, terminal = (sum((row[index] for row in rows), Fraction(0)) for index in range(4))
        result = dict(total=len(group), admitted=sum(job["completion"] is not None for job in group),
            waiting_area=w, inflight_area=b, common_area=area, terminal_cost=terminal, cost=area + terminal)
        if overflow:
            result["known_completion_residual"] = sum((max(Fraction(0), job["completion"] - horizon)
                for job in group if job["completion"] is not None), Fraction(0))
            result["pending_waiting_accrual"] = sum((max(Fraction(0), partial_end - horizon)
                for job in group if job["completion"] is None), Fraction(0))
        if not overflow and result["cost"] != sum((job["completion"] - max(Fraction(0), job["birth"])
                for job in group), Fraction(0)):
            raise ValueError("cached common area and native terminal do not reconstruct completion sums")
        return result

    sources = ("visible", "prefix", "background", "committed")
    population = group_result(jobs)
    if (population["common_area"], population["waiting_area"], population["inflight_area"]) != (
            common_area, waiting_area, inflight_area):
        raise ValueError("restoring H queues changed the cached window costs")
    population["remaining"] = remaining
    population["sources"] = {source: group_result([job for job in jobs if job["source"] == source])
        for source in sources}
    population["per_coordinate"] = tuple(dict(coordinate=coordinate,
        **group_result([job for job in jobs if job["coordinate"] == coordinate]),
        sources={source: group_result([job for job in jobs
            if job["coordinate"] == coordinate and job["source"] == source]) for source in sources})
        for coordinate in range(len(waiting)))
    population["jobs"] = tuple(jobs)
    after_suffix = (len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    incremental = dict(zip(("background_events", "random_draws", "rate_reads", "seed_initializations"),
        (after - before for after, before in zip(after_suffix, before_suffix, strict=True)), strict=True))
    incremental.update(dispatch_work)
    incremental.update(native_controller_calls=calls, native_idle_calls=idle_calls,
        competing_admissions=competing_admissions)
    announce(force=True)
    return dict(cost=population["cost"], common_area=common_area, waiting_area=waiting_area,
        inflight_area=inflight_area, terminal_cost=population["terminal_cost"],
        complete=not overflow and not remaining, overflow=overflow, horizon=horizon,
        partial_terminal_observation_end=partial_end if overflow else None, population=population,
        calls=calls, idle_calls=idle_calls, terminal_calls=calls, terminal_idle_calls=idle_calls,
        logical_work=dict(logical_work), proposal_status=dict(statuses), dispatch_work=dict(dispatch_work),
        incremental_tail_work=incremental, final_lane_ends=tuple(lane_ends),
        final_queue=tuple(map(len, waiting)), final_counts=tuple(counts),
        first_call_context=first_call_context, first_native_decision=first_native_decision,
        background_events=tuple(suffix.events), suffix_work=after_suffix,
        worker_cpu_s=process_time() - cpu, worker_wall_s=perf_counter() - wall)


def evaluate_cached(options, planning_seed, cached_result, origin_now, *, workers=6,
        sample_count=8, event_cap=128, progress=None):
    """Restore one original r1 JSON result and score its native finite-policy tail."""
    cpu, wall = process_time(), perf_counter()
    old, environment = cached_result["flow"], cached_result["environment"]
    if not old["complete"] or old["overflow"] or environment["event_cap"] != event_cap:
        raise ValueError("a complete fixed-H source with the unchanged cap is required")
    horizon = _fraction(old["horizon"])
    if horizon != _fraction(environment["horizon"]) or horizon != 2 * Fraction.from_float(environment["window_s"]):
        raise ValueError("cached fixed common horizon changed")
    expected_seed = f"{planning_seed}:background:{environment['anchor_counter']}:{cached_result['sample_index']}"
    if environment["seed"] != expected_seed:
        raise ValueError("original planning seed or global sample background identity changed")
    suffix = _LazyBackgroundArrivals(environment["rates"], environment["window_s"],
        environment["seed"], event_cap, len(environment["prefix_events"]))
    if not suffix.release_through(horizon):
        raise ValueError("the completed cached H suffix cannot be regenerated")
    cached_events = tuple((_fraction(event[0]), event[1]) for event in old["background_events"])
    regenerated_work = (len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    if tuple(suffix.events) != cached_events or regenerated_work != tuple(old["suffix_work"]):
        raise ValueError("cached H background events or native sampling work changed")
    jobs = [dict(job, id=tuple(job["id"]), birth=_fraction(job["birth"]),
        start=_fraction(job["start"]) if job["start"] is not None else None,
        completion=_fraction(job["completion"]) if job["completion"] is not None else None)
        for job in old["population"]["jobs"]]
    terminal_state = dict(old["terminal_state"],
        lane_ends=tuple(_fraction(end) for end in old["terminal_state"]["lane_ends"]))
    if len(old["final_tree"]["planners"]) != 3:
        raise ValueError("the complete native three-controller H state is required")
    constructor_cpu, constructor_wall = process_time(), perf_counter()
    with native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
            options, planning_seed, sample_count, event_cap, workers=workers) as planner:
        constructor_cpu, constructor_wall = process_time() - constructor_cpu, perf_counter() - constructor_wall
        restore_work = restore_planner_tree(planner, old["final_tree"])
        initial_tree = capture_planner_tree(planner, restore_work)
        before_actual = planner.reuse_work
        flow = _simulate_native_terminal(planner, jobs, terminal_state, suffix,
            origin=Fraction.from_float(float(origin_now)), horizon=horizon,
            common_area=_fraction(old["common_area"]), waiting_area=_fraction(old["waiting_area"]),
            inflight_area=_fraction(old["inflight_area"]), progress=progress)
        if planner.decision_counter != old["final_tree"]["planners"][0]["decision_counter"] + flow["calls"]:
            raise ValueError("native terminal failed to consume every dispatch including idle")
        after_actual = planner.reuse_work
        actual_work = {group: {name: value - before_actual[group].get(name, 0)
            for name, value in values.items()} for group, values in after_actual.items()}
        final_tree = capture_planner_tree(planner, restore_work)
        flow.update(past_flow_constant=_fraction(old["past_flow_constant"]),
            regeneration_work=dict(zip(("background_events", "random_draws", "rate_reads", "seed_initializations"),
                regenerated_work, strict=True)),
            reused_window_work=dict(native_controller_calls=old["calls"], native_idle_calls=old["idle_calls"],
                logical_work=old["logical_work"], proposal_status=old["proposal_status"]),
            restore_work=dict(restore_work), initial_tree=initial_tree, final_tree=final_tree,
            actual_work=actual_work, preprocessing=dict(planner.preprocessing_work),
            controller_constructors=3, constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
            total_cpu_s=process_time() - cpu, total_wall_s=perf_counter() - wall)
    return dict(root=cached_result["root"], sample_index=cached_result["sample_index"],
        environment=environment, flow=flow, complete=flow["complete"],
        scored_population=tuple({field: job[field] for field in ("id", "coordinate", "birth", "source")}
            for job in flow["population"]["jobs"]))
