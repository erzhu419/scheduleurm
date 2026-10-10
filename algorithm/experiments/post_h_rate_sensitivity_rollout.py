"""Change external post-H rates while retaining the cached native H state."""
from fractions import Fraction
from math import isfinite
from time import perf_counter, process_time

from algorithm import parallel_root_lane_rollout as native
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.competing_terminal_flow_rollout import _fraction
from algorithm.experiments.native_terminal_flow_rollout import _simulate_native_terminal
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)


def switch_post_h_rates(suffix, horizon, post_h_rates):
    """Retain each pending exponential quantile and the existing RNG at H."""
    old_rates, new_rates = tuple(suffix.rates), tuple(post_h_rates)
    old_pending = tuple(suffix.next_times)
    if (suffix.rng is None or suffix.overflow or len(new_rates) != len(old_rates)
            or len(old_pending) != len(old_rates)):
        raise ValueError("an initialized complete H suffix and matching rates are required")
    if any(not isfinite(rate) or rate <= 0 for rate in old_rates + new_rates):
        raise ValueError("the frozen diagnostic requires finite positive rates")
    if any(not isfinite(time) or Fraction.from_float(time) <= horizon for time in old_pending):
        raise ValueError("each pending arrival must be strictly after the frozen H")
    changed = 0
    h = float(horizon)
    for coordinate, (old, new, pending) in enumerate(zip(old_rates, new_rates, old_pending, strict=True)):
        if old != new:
            suffix.next_times[coordinate] = h + (pending - h) * (old / new)
            changed += 1
    suffix.rates = new_rates
    work = dict(rate_coordinate_reads=2 * len(old_rates),
        pending_time_coordinate_reads=len(old_pending), rate_coordinate_writes=len(new_rates),
        residual_subtractions=changed, rate_ratio_divisions=changed,
        residual_multiplications=changed, horizon_additions=changed,
        pending_time_writes=changed, random_draws=0, seed_initializations=0)
    return dict(horizon=horizon, old_rates=old_rates, new_rates=new_rates,
        old_pending_times=old_pending, new_pending_times=tuple(suffix.next_times), work=work)


def evaluate_cached(options, planning_seed, cached_result, origin_now, *, post_h_rates,
        workers=6, sample_count=8, event_cap=128, progress=None):
    """Score the original H roster under native control and new external rates."""
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
    rate_switch = switch_post_h_rates(suffix, horizon, post_h_rates)
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
            rate_switch=rate_switch, rate_switch_work=rate_switch["work"],
            restore_work=dict(restore_work), initial_tree=initial_tree, final_tree=final_tree,
            actual_work=actual_work, preprocessing=dict(planner.preprocessing_work),
            controller_constructors=3, constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
            total_cpu_s=process_time() - cpu, total_wall_s=perf_counter() - wall)
    return dict(root=cached_result["root"], sample_index=cached_result["sample_index"],
        environment=environment, post_h_rates=rate_switch["new_rates"], flow=flow, complete=flow["complete"],
        scored_population=tuple({field: job[field] for field in ("id", "coordinate", "birth", "source")}
            for job in flow["population"]["jobs"]))
