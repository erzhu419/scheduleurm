"""Reflect the fixed H newborn roster and replay its complete native policy.

The external post-H stream retains the original clocks and RNG. The shared H
simulator also performs a fixed MW drain; its assignments are undone and that
extra diagnostic work is charged separately from the native dispatches.
"""
from fractions import Fraction
from time import perf_counter, process_time

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.common_horizon_flow_rollout import _simulate_flow
from algorithm.experiments.native_terminal_flow_rollout import _simulate_native_terminal
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)


def _time(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(*value) if isinstance(value, (list, tuple)) else Fraction.from_float(float(value))


def reflect_roster(jobs, horizon, window, *, reflect=True):
    """Copy stable identities; change only newborn birth times and time sources."""
    horizon, window = _time(horizon), _time(window)
    roster = []
    for original in jobs:
        job = dict(original, id=tuple(original["id"]), birth=_time(original["birth"]))
        for field in ("start", "completion"):
            if field in job and job[field] is not None:
                job[field] = _time(job[field])
        if job["source"] in ("prefix", "background"):
            if reflect:
                job["birth"] = horizon - job["birth"]
            job["source"] = "prefix" if job["birth"] <= window else "background"
        roster.append(job)
    return tuple(roster)


def simulate_within_h(planner, context, root, roster, *, initial_inflight=(), progress=None):
    """Replay the H roster, retaining stable FIFO identities and H commitments."""
    horizon = 2 * Fraction.from_float(float(planner.maximum_duration))
    newborn = sorted((job for job in roster if job["source"] in ("prefix", "background")),
        key=lambda job: (job["birth"], job["id"]))
    events = tuple((job["birth"], job["coordinate"]) for job in newborn)
    # All known H births are in events; this suffix releases nothing through H.
    empty = _LazyBackgroundArrivals((0.,) * len(context["queue"]), float(horizon),
        "fixed_H_roster", len(events), 0)

    def window_progress(fields):
        if fields["phase"] in ("complete", "overflow"):
            fields = dict(fields, phase="H_window_" + fields["phase"])
        progress(fields)

    replay = _simulate_flow(planner, context, root, events, empty,
        initial_inflight=initial_inflight, progress=window_progress if progress is not None else None)
    replacements = {("prefix", index): job for index, job in enumerate(newborn)}

    def identity(value):
        key = tuple(value)
        return replacements[key]["id"] if key in replacements else key

    terminal_state = dict(replay["terminal_state"],
        waiting_ids=tuple(tuple(identity(value) for value in group)
            for group in replay["terminal_state"]["waiting_ids"]),
        unfinished_ids=tuple(identity(value) for value in replay["terminal_state"]["unfinished_ids"]))
    waiting_ids = {value for group in terminal_state["waiting_ids"] for value in group}
    jobs = []
    for original in replay["population"]["jobs"]:
        job = dict(original)
        replacement = replacements.get(tuple(job["id"]))
        if replacement is not None:
            job.update(id=replacement["id"], source=replacement["source"])
        if job["id"] in waiting_ids:
            job["start"] = job["completion"] = None
        jobs.append(job)
    by_id = {job["id"]: job for job in jobs}
    jobs = [by_id[tuple(job["id"])] for job in roster]
    dispatch = dict(replay["dispatch_work"])
    drain = dict(replay["terminal_work"], fixed_mw_calls=replay["terminal_calls"],
        fifo_identity_admissions=len(waiting_ids))
    dispatch["fifo_identity_admissions"] -= drain["fifo_identity_admissions"]
    return dict({field: replay[field] for field in (
        "common_area", "waiting_area", "inflight_area", "horizon", "area_end", "overflow",
        "calls", "idle_calls", "logical_work", "proposal_status", "first_call_context",
        "first_native_decision", "worker_cpu_s", "worker_wall_s")},
        terminal_state=terminal_state, jobs=tuple(jobs), dispatch_work=dispatch,
        diagnostic_drain_work=drain)


def _difference(after, before):
    return {group: {name: value - before.get(group, {}).get(name, 0)
        for name, value in values.items()} for group, values in after.items()}


def evaluate_cached(options, planning_seed, cached_result, context, after_tree, *,
        initial_inflight=(), workers=6, sample_count=8, event_cap=128, progress=None, reflect=True):
    """Replay original or mirrored H times, then the unchanged native tail."""
    cpu, wall = process_time(), perf_counter()
    old, environment = cached_result["flow"], cached_result["environment"]
    if not old["complete"] or old["overflow"] or environment["event_cap"] != event_cap:
        raise ValueError("a complete original H source and unchanged cap are required")
    horizon, window = _time(old["horizon"]), _time(environment["window_s"])
    if horizon != _time(environment["horizon"]) or horizon != 2 * window:
        raise ValueError("original common H changed")
    seed = f"{planning_seed}:background:{environment['anchor_counter']}:{cached_result['sample_index']}"
    if environment["seed"] != seed:
        raise ValueError("original planning seed or global sample identity changed")
    if len(after_tree["planners"]) != 3:
        raise ValueError("the complete consumed three-controller anchor is required")
    anchor = after_tree["planners"][0]
    rates = tuple(count / context["now"] if context["now"] > 0 else 0. for count in context["counts"])
    if (anchor["decision_counter"] != environment["anchor_counter"] + 1
            or tuple(anchor["arrival_rates"]) != rates
            or tuple(anchor["observed_counts"]) != tuple(context["counts"])
            or tuple(environment["rates"]) != rates):
        raise ValueError("consumed anchor counters, rates or physical context changed")
    suffix = _LazyBackgroundArrivals(environment["rates"], environment["window_s"],
        seed, event_cap, len(environment["prefix_events"]))
    if not suffix.release_through(horizon):
        raise ValueError("the completed original H suffix cannot be regenerated")
    regenerated = (len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
    expected = tuple((_time(event[0]), event[1]) for event in old["background_events"])
    if tuple(suffix.events) != expected or regenerated != tuple(old["suffix_work"]):
        raise ValueError("original H events or sampling work changed")
    original_h_stream = dict(events=tuple(suffix.events), pending_times=tuple(suffix.next_times))
    reflection_cpu, reflection_wall = process_time(), perf_counter()
    roster = reflect_roster(old["population"]["jobs"], horizon, window, reflect=reflect)
    reflection_cpu, reflection_wall = process_time() - reflection_cpu, perf_counter() - reflection_wall
    newborn_count = sum(job["source"] in ("prefix", "background") for job in roster)
    reflection_work = dict(roster_job_reads=len(roster), roster_job_copies=len(roster),
        newborn_birth_subtractions=newborn_count if reflect else 0,
        temporal_source_comparisons=newborn_count, newborn_sort_inputs=newborn_count,
        newborn_sort_key_reads=2 * newborn_count, newborn_sorts=1,
        stable_roster_order_lookups=len(roster),
        random_draws=0, seed_initializations=0, new_anchor_proposals=0)
    constructor_cpu, constructor_wall = process_time(), perf_counter()
    with native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
            options, planning_seed, sample_count, event_cap, workers=workers) as planner:
        constructor_cpu, constructor_wall = process_time() - constructor_cpu, perf_counter() - constructor_wall
        if Fraction.from_float(float(planner.maximum_duration)) != window:
            raise ValueError("registered physical catalogue changed the original window")
        if any(need > count for need, count in zip(planner.requirements[cached_result["root"]],
                context["queue"], strict=True)):
            raise ValueError("registered physical root is infeasible")
        restore_work = restore_planner_tree(planner, after_tree)
        initial_tree = capture_planner_tree(planner, restore_work)
        before_actual = planner.reuse_work
        h_flow = simulate_within_h(planner, context, cached_result["root"], roster,
            initial_inflight=initial_inflight, progress=progress)
        if h_flow["overflow"]:
            raise ValueError("the fixed complete H roster unexpectedly overflowed")
        if planner.decision_counter != anchor["decision_counter"] + h_flow["calls"]:
            raise ValueError("native H replay did not consume every dispatch including idle")
        h_tree = capture_planner_tree(planner, restore_work)
        h_actual = planner.reuse_work
        flow = _simulate_native_terminal(planner, h_flow["jobs"], h_flow["terminal_state"], suffix,
            origin=_time(context["now"]), horizon=horizon, common_area=h_flow["common_area"],
            waiting_area=h_flow["waiting_area"], inflight_area=h_flow["inflight_area"], progress=progress)
        if planner.decision_counter != h_tree["planners"][0]["decision_counter"] + flow["calls"]:
            raise ValueError("native terminal did not consume every dispatch including idle")
        after_actual = planner.reuse_work
        flow.update(past_flow_constant=_time(old["past_flow_constant"]),
            original_h_stream=original_h_stream, window_flow=h_flow,
            regeneration_work=dict(zip(("background_events", "random_draws", "rate_reads", "seed_initializations"),
                regenerated, strict=True)), reflection_work=reflection_work,
            reflection_cpu_s=reflection_cpu, reflection_wall_s=reflection_wall,
            diagnostic_drain_work=h_flow["diagnostic_drain_work"],
            restore_work=dict(restore_work), initial_tree=initial_tree, h_tree=h_tree,
            final_tree=capture_planner_tree(planner, restore_work),
            actual_work=_difference(after_actual, before_actual),
            h_actual_work=_difference(h_actual, before_actual),
            terminal_actual_work=_difference(after_actual, h_actual),
            preprocessing=dict(planner.preprocessing_work), controller_constructors=3,
            constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
            total_cpu_s=process_time() - cpu, total_wall_s=perf_counter() - wall)
        flow["restore_work"] = dict(restore_work)
    return dict(root=cached_result["root"], sample_index=cached_result["sample_index"],
        environment=environment, mirror=reflect, flow=flow, complete=flow["complete"],
        scored_population=tuple({field: job[field] for field in ("id", "coordinate", "birth", "source")}
            for job in flow["population"]["jobs"]))
