"""Price every unfinished job over one common window and a fixed MW drain.

The finite controller consumes its complete native proposals before the common
horizon. Births use the original prefix and globally indexed background stream.
At the horizon, already committed completions and reserved lane release times
remain distinct; fixed native MaxWeight clears the population without new births.
"""
from collections import Counter, deque
from fractions import Fraction
from time import perf_counter, process_time

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm import parallel_root_lane_rollout as native
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, STATUS
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


def _exact(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def _simulate_flow(planner, context, root, events, suffix, *, initial_inflight=(), progress=None):
    """Simulate one root from an already restored after-anchor controller.

    ``initial_inflight`` contains committed jobs with absolute birth/start and
    completion clocks. Their resource release times come only from ``available``.
    This separate entry point permits small, independent hand-computed tests.
    """
    cpu, wall = process_time(), perf_counter()
    origin = _exact(context["now"])
    horizon = 2 * _exact(planner.maximum_duration)
    lane_ends = [max(Fraction(0), _exact(end) - origin) for end in context["available"]]
    waiting = [deque() for _ in context["queue"]]
    jobs, prefix = [], []
    counts = list(context["counts"])
    logical_work, statuses, dispatch_work, terminal_work = (Counter() for _ in range(4))
    calls = idle_calls = terminal_calls = 0
    cursor = background_cursor = 0
    first_call_context = first_native_decision = None
    last_progress = wall - 30

    def entry(identity, coordinate, birth, source, start=None, completion=None):
        job = dict(id=identity, coordinate=coordinate, birth=birth, source=source,
            start=start, completion=completion)
        jobs.append(job)
        return job

    for coordinate, ages in enumerate(context["ages"]):
        for index, age in enumerate(ages):
            waiting[coordinate].append(entry(("visible", coordinate, index), coordinate,
                -_exact(age), "visible"))
    for index, initial in enumerate(initial_inflight):
        completion = _exact(initial["completion"]) - origin
        if completion <= 0:
            raise ValueError("initial committed population must still be unfinished at the anchor")
        entry(initial.get("id", ("committed", index)), initial["coordinate"],
            _exact(initial.get("birth", context["now"])) - origin, "committed",
            _exact(initial.get("start", context["now"])) - origin, completion)
    for index, (birth, coordinate) in enumerate(events):
        prefix.append(entry(("prefix", index), coordinate, birth, "prefix"))

    def announce(phase, *, force=False, proposal_wall_s=None):
        nonlocal last_progress
        now = perf_counter()
        if progress is not None and (force or now - last_progress >= 30):
            progress(dict(phase=phase, root=root, calls=calls, idle_calls=idle_calls,
                terminal_calls=terminal_calls, waiting_jobs=sum(map(len, waiting)),
                known_jobs=len(jobs), background_events=len(suffix.events),
                elapsed_wall_s=now - wall, elapsed_cpu_s=process_time() - cpu,
                proposal_wall_s=proposal_wall_s))
            last_progress = now

    def admit(index, start):
        for coordinate, offsets in enumerate(planner.sorted_completion_offsets[index]):
            for offset in offsets:
                job = waiting[coordinate].popleft()
                job["start"], job["completion"] = start, start + offset
                dispatch_work["fifo_identity_admissions"] += 1

    def release(cutoff):
        nonlocal cursor, background_cursor
        while cursor < len(events) and events[cursor][0] <= cutoff:
            job = prefix[cursor]
            waiting[job["coordinate"]].append(job)
            counts[job["coordinate"]] += 1
            dispatch_work["observed_arrival_counter_updates"] += 1
            cursor += 1
        complete = suffix.release_through(cutoff)
        while background_cursor < len(suffix.events):
            birth, coordinate = suffix.events[background_cursor]
            if birth > cutoff:
                break
            waiting[coordinate].append(entry(("background", background_cursor), coordinate,
                birth, "background"))
            counts[coordinate] += 1
            dispatch_work["observed_arrival_counter_updates"] += 1
            background_cursor += 1
        return complete

    lane_ends[context["lane"]] = planner.durations[root]
    admit(root, Fraction(0))
    overflow = False
    area_end = horizon
    announce("native", force=True)
    while True:
        lane = min(range(len(lane_ends)), key=lambda index: (lane_ends[index], index))
        start = lane_ends[lane]
        cutoff = min(start, horizon)
        if not release(cutoff):
            overflow = True
            area_end = suffix.events[-1][0]
            break
        if start >= horizon:
            break
        queue = tuple(len(group) for group in waiting)
        ages = tuple(tuple(float(start - job["birth"]) for job in group) for group in waiting)
        current = (planner.options, queue, tuple(float(origin + end) for end in lane_ends),
            float(origin + start), lane, ages, tuple(counts))
        dispatch_work["age_context_job_reads"] += sum(queue)
        dispatch_work["observed_arrival_context_coordinate_reads"] += len(counts)
        dispatch_work["lane_context_reads"] += len(lane_ends)
        announce("native")
        proposal_begin = perf_counter()
        decision = planner(*current)
        calls += 1
        logical_work.update({name: getattr(decision, name) for name in FEES + TWO_STEP_WORK})
        for group in (decision.inner_current_work, decision.inner_current_fallbacks):
            logical_work.update({"inner_current_" + name: value for name, value in group.items()})
        statuses.update({name: getattr(decision, name) for name in STATUS})
        if calls == 1:
            first_call_context, first_native_decision = current[1:], decision
        chosen, = decision.selection.option_indices
        if chosen is None:
            idle_calls += 1
            lane_ends[lane] = start + 1
        else:
            admit(chosen, start)
            lane_ends[lane] = start + planner.durations[chosen]
        announce("native", proposal_wall_s=perf_counter() - proposal_begin)

    # Capture the actual H state before the fixed terminal controller admits jobs.
    terminal_state = dict(queue=tuple(map(len, waiting)),
        waiting_ids=tuple(tuple(job["id"] for job in group) for group in waiting),
        lane_ends=tuple(lane_ends), counts=tuple(counts),
        unfinished_ids=tuple(job["id"] for job in jobs
            if job["birth"] <= area_end and
            (job["completion"] is None or job["completion"] > area_end)))
    if not overflow:
        lane_ends = [max(horizon, end) for end in lane_ends]
        announce("terminal_mw", force=True)
        while any(waiting):
            lane = min(range(len(lane_ends)), key=lambda index: (lane_ends[index], index))
            start = lane_ends[lane]
            queue = tuple(map(len, waiting))
            chosen, feasible = planner._best_index(queue)
            terminal_calls += 1
            terminal_work["root_option_probes"] += len(planner.options)
            terminal_work["feasible_root_options"] += feasible
            terminal_work["queue_coordinate_reads"] += len(queue)
            if chosen is None:
                raise ValueError("native MW cannot clear the registered FIFO population")
            admit(chosen, start)
            lane_ends[lane] = start + planner.durations[chosen]
            announce("terminal_mw")

    def components(job):
        begin = max(Fraction(0), job["birth"])
        if begin > area_end:
            return (Fraction(0),) * 4
        start = job["start"]
        completion = job["completion"]
        waiting_area = max(Fraction(0), min(area_end, start if start is not None else area_end) - begin)
        inflight_area = (max(Fraction(0), min(area_end, completion) - max(Fraction(0), start))
            if completion is not None else Fraction(0))
        residual = max(Fraction(0), completion - horizon) if not overflow else Fraction(0)
        return waiting_area, inflight_area, waiting_area + inflight_area, residual

    def group_result(group):
        values = [components(job) for job in group]
        waiting_area, inflight_area, common_area, terminal_cost = (
            sum((row[index] for row in values), Fraction(0)) for index in range(4))
        result = dict(total=len(group), admitted=sum(job["completion"] is not None for job in group),
            waiting_area=waiting_area, inflight_area=inflight_area,
            common_area=common_area, terminal_cost=terminal_cost, cost=common_area + terminal_cost)
        if not overflow and result["cost"] != sum(
                (job["completion"] - max(Fraction(0), job["birth"]) for job in group), Fraction(0)):
            raise ValueError("common area and terminal residual do not reconstruct completion sums")
        return result

    sources = ("visible", "prefix", "background", "committed")
    population = group_result(jobs)
    population["remaining"] = population["total"] - population["admitted"]
    population["sources"] = {source: group_result([job for job in jobs if job["source"] == source])
        for source in sources}
    population["per_coordinate"] = tuple(dict(coordinate=coordinate,
        **group_result([job for job in jobs if job["coordinate"] == coordinate]),
        sources={source: group_result([job for job in jobs
            if job["coordinate"] == coordinate and job["source"] == source]) for source in sources})
        for coordinate in range(len(waiting)))
    population["jobs"] = tuple(jobs)
    past = sum((max(Fraction(0), -job["birth"]) for job in jobs
        if job["source"] in ("visible", "committed")), Fraction(0))
    result = dict(cost=population["cost"], common_area=population["common_area"],
        waiting_area=population["waiting_area"], inflight_area=population["inflight_area"],
        terminal_cost=population["terminal_cost"], past_flow_constant=past,
        horizon=horizon, area_end=area_end, overflow=overflow,
        complete=not overflow and not population["remaining"], population=population,
        calls=calls, idle_calls=idle_calls, terminal_calls=terminal_calls,
        logical_work=dict(logical_work), proposal_status=dict(statuses),
        dispatch_work=dict(dispatch_work), terminal_work=dict(terminal_work),
        terminal_state=terminal_state, final_lane_ends=tuple(lane_ends),
        first_call_context=first_call_context, first_native_decision=first_native_decision,
        background_events=tuple(suffix.events),
        suffix_work=(len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations),
        worker_cpu_s=process_time() - cpu, worker_wall_s=perf_counter() - wall)
    announce("complete" if result["complete"] else "overflow", force=True)
    return result


def _score_flow(planner, context, root, events, suffix, after_tree, *, initial_inflight=(), progress=None):
    work = restore_planner_tree(planner, after_tree)
    initial_tree = capture_planner_tree(planner, work)
    result = _simulate_flow(planner, context, root, events, suffix,
        initial_inflight=initial_inflight, progress=progress)
    if planner.decision_counter != after_tree["planners"][0]["decision_counter"] + result["calls"]:
        raise ValueError("finite controller did not consume every real dispatch before the horizon")
    result.update(restore_work=dict(work), initial_tree=initial_tree,
        final_tree=capture_planner_tree(planner, work), actual_work=planner.reuse_work)
    result["restore_work"] = dict(work)
    return result


def evaluate_root_sample(options, planning_seed, paths, before_tree, after_tree, context,
        root, sample_index, *, initial_inflight=(), sample_count=8, event_cap=128,
        workers=6, progress=None):
    """Evaluate one registered root/global sample without changing its risk gate."""
    if len(paths) != sample_count or not 0 <= sample_index < sample_count:
        raise ValueError("the original full sample roster and global index are required")
    if len(before_tree["planners"]) != 3 or len(after_tree["planners"]) != 3:
        raise ValueError("finite before/after trees must contain all three controllers")
    anchor = before_tree["planners"][0]["decision_counter"]
    if after_tree["planners"][0]["decision_counter"] != anchor + 1:
        raise ValueError("after tree must have consumed one complete native anchor proposal")
    path = paths[sample_index]
    if isinstance(path, PlanningArrivals):
        if path.overflow:
            raise ValueError("the registered original prefix is unavailable")
        path = path.events
    events = tuple((_exact(event[0]), event[1]) for event in path)
    queue, counts = tuple(context["queue"]), tuple(context["counts"])
    if len(context["ages"]) != len(queue) or any(len(group) != count
            for group, count in zip(context["ages"], queue, strict=True)):
        raise ValueError("visible FIFO population differs from the supplied queue")
    rates = tuple(count / context["now"] if context["now"] > 0 else 0. for count in counts)
    after = after_tree["planners"][0]
    if tuple(after["arrival_rates"]) != rates or tuple(after["observed_counts"]) != counts:
        raise ValueError("after proposal rates/counts differ from the anchor context")
    constructor_cpu, constructor_wall = process_time(), perf_counter()
    with native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
            options, planning_seed, sample_count, event_cap, workers=workers) as planner:
        constructor_cpu, constructor_wall = process_time() - constructor_cpu, perf_counter() - constructor_wall
        if any(need > count for need, count in zip(planner.requirements[root], queue, strict=True)):
            raise ValueError("registered physical root is infeasible")
        seed = f"{planning_seed}:background:{anchor}:{sample_index}"
        suffix = _LazyBackgroundArrivals(rates, planner.maximum_duration, seed, event_cap, len(events))
        flow = _score_flow(planner, context, root, events, suffix, after_tree,
            initial_inflight=initial_inflight, progress=progress)
        flow.update(constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
            preprocessing=dict(planner.preprocessing_work), controller_constructors=3)
        environment = dict(rates=rates, window_s=planner.maximum_duration, horizon=flow["horizon"],
            seed=seed, event_cap=event_cap, sample_index=sample_index,
            anchor_counter=anchor, prefix_events=events)
    return dict(root=root, sample_index=sample_index, environment=environment,
        flow=flow, complete=flow["complete"])
