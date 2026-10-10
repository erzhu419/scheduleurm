"""Compare native two-current and free finite continuation on one fixed sample.

The synthetic environment keeps the anchor's rates and background seed. The
free controller starts after the complete real finite proposal, then consumes
every virtual dispatch, including empty MW fallbacks. Only the original
visible/prefix tagged population contributes cost. No real tape is replayed.
"""
import ast
from collections import Counter, deque
from fractions import Fraction
import inspect
import textwrap
from time import perf_counter, process_time
from types import FunctionType

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm import parallel_root_lane_rollout as native
from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, STATUS
from algorithm.experiments.persistent_continuation_cross_selector import (
    capture_planner_tree, restore_planner_tree,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK


def _exact(value):
    return Fraction.from_float(float(value))


class _Population:
    """Track stable FIFO identities and exact costs alongside native admissions."""
    def __init__(self, queue, ages, events):
        self.waiting = [deque() for _ in queue]
        self.population, self.completed = [], []
        self.groups = {(coordinate, source): dict(tagged_total=0, evaluations=0,
            cost=Fraction(0), squared=Fraction(0))
            for coordinate in range(len(queue)) for source in ("visible", "prefix")}
        self.work = Counter()
        for coordinate, group in enumerate(ages):
            for index, age in enumerate(group):
                entry = self._entry(("visible", coordinate, index), coordinate, -_exact(age), "visible")
                self.waiting[coordinate].append(entry)
        for index, (birth, coordinate) in enumerate(events):
            self._entry(("prefix", index), coordinate, birth, "prefix")

    def _entry(self, identity, coordinate, birth, source):
        entry = dict(id=identity, coordinate=coordinate, birth=birth, source=source)
        self.population.append(entry)
        self.groups[coordinate, source]["tagged_total"] += 1
        self.work["tagged_identity_initializations"] += 1
        return entry

    def append(self, coordinate, birth, tagged, index):
        identity = ("prefix", index) if tagged else ("background", index)
        self.waiting[coordinate].append(dict(id=identity, coordinate=coordinate, birth=birth,
            source="prefix" if tagged else "background"))
        self.work["fifo_identity_appends"] += 1

    def admit(self, coordinate, start, offset, birth, tagged):
        entry = self.waiting[coordinate].popleft()
        if entry["birth"] != birth or (entry["source"] != "background") != tagged:
            raise ValueError("population observer differs from the native FIFO admission")
        self.work["fifo_identity_admissions"] += 1
        if tagged:
            flow = start + offset - birth
            group = self.groups[coordinate, entry["source"]]
            group["evaluations"] += 1
            group["cost"] += flow
            group["squared"] += flow * flow
            self.completed.append(entry["id"])
            self.work["tagged_cost_evaluations"] += 1

    def result(self, cost, squared):
        def total(groups):
            return {name: sum(group[name] for group in groups)
                for name in ("tagged_total", "evaluations", "cost", "squared")}
        per_coordinate = tuple(dict(coordinate=coordinate,
            **total([self.groups[coordinate, source] for source in ("visible", "prefix")]),
            sources={source: dict(self.groups[coordinate, source]) for source in ("visible", "prefix")})
            for coordinate in range(len(self.waiting)))
        totals = total(list(self.groups.values()))
        if totals["cost"] != cost or totals["squared"] != squared:
            raise ValueError("exact population components do not reconstruct the native costs")
        return dict(**totals, tagged_remaining=totals["tagged_total"] - totals["evaluations"],
            tagged_population=tuple(self.population), completed_tagged_ids=tuple(self.completed),
            per_coordinate=per_coordinate,
            sources={source: total([self.groups[coordinate, source]
                for coordinate in range(len(self.waiting))]) for source in ("visible", "prefix")},
            observer_work=dict(self.work))


def _instrumented_native_code():
    """Add identity bookkeeping at existing FIFO append/pop sites only."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(native.score_single_root)))

    class Instrument(ast.NodeTransformer):
        admissions = appends = 0

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [getattr(item, "id", None) for item in node.targets[0].elts] == ["birth", "tagged"]):
                self.admissions += 1
                return [node, ast.parse("_population.admit(coordinate, start, offset, birth, tagged)").body[0]]
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "append" and isinstance(call.func.value, ast.Subscript)
                    and isinstance(call.func.value.value, ast.Name) and call.func.value.value.id == "waiting"
                    and len(call.args) == 1 and isinstance(call.args[0], ast.Tuple)):
                tagged = call.args[0].elts[1].value
                self.appends += 1
                index = "cursor" if tagged else "background_cursor"
                hook = f"_population.append(coordinate, birth, {tagged!r}, {index})"
                return [node, ast.parse(hook).body[0]]
            return node

    instrument = Instrument()
    tree = instrument.visit(tree)
    if (instrument.admissions, instrument.appends) != (1, 2):
        raise ValueError("native population recording sites changed")
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, __file__, "exec"), native.score_single_root.__globals__, namespace)
    return namespace["score_single_root"].__code__


_NATIVE_CODE = _instrumented_native_code()


def _tree_semantic(saved):
    return (saved["rng_state"], saved["decision_counter"], tuple(saved["arrival_rates"]),
        tuple(saved["observed_counts"]) if saved["observed_counts"] is not None else None)


def _background_result(suffix):
    return dict(background_events=tuple(suffix.events),
        suffix_work=(len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations))


def _score_approx(options, seed, samples, cap, payload, population):
    """Use the unchanged physical native worker, with FIFO recording attached."""
    suffixes = []

    def suffix(*args):
        result = _LazyBackgroundArrivals(*args)
        suffixes.append(result)
        return result

    scope = {**native.score_single_root.__globals__, "_population": population,
        "_LazyBackgroundArrivals": suffix}
    scorer = FunctionType(_NATIVE_CODE, scope, "score_single_root")
    worker_scope = {**native._run_root.__globals__, "_worker_config": (options, seed, samples, cap),
        "_worker_planner": None, "score_single_root": scorer}
    result = FunctionType(native._run_root.__code__, worker_scope, "_run_root")(payload)
    result["population"] = population.result(result["cost"], result["squared"])
    result["background_events"] = tuple(suffixes[0].events)
    result["complete"] = not result["overflow"] and not result["population"]["tagged_remaining"]
    return result


def _score_free(planner, context, root, events, suffix, after_tree, population):
    cpu, wall = process_time(), perf_counter()
    restore_work = restore_planner_tree(planner, after_tree)
    initial_tree = capture_planner_tree(planner, restore_work)
    queue, ages = context["queue"], context["ages"]
    origin = _exact(context["now"])
    lane_ends = [max(Fraction(0), _exact(end) - origin) for end in context["available"]]
    waiting = [deque((-_exact(age), True) for age in group) for group in ages]
    counts = list(context["counts"])
    lane_ends[context["lane"]] = planner.durations[root]
    tagged_remaining = sum(queue) + len(events)
    cost = squared = Fraction(0)
    calls = idle_calls = evaluations = admitted_reads = tag_reads = background_admissions = 0
    cursor = background_cursor = 0
    logical_work, statuses, dispatch_work = Counter(), Counter(), Counter()
    first_call_context = first_native_decision = None
    overflow = False

    def admit(index, start):
        nonlocal tagged_remaining, cost, squared, evaluations, admitted_reads, tag_reads, background_admissions
        for coordinate, offsets in enumerate(planner.sorted_completion_offsets[index]):
            for offset in offsets:
                birth, tagged = waiting[coordinate].popleft()
                admitted_reads += 1
                tag_reads += 1
                population.admit(coordinate, start, offset, birth, tagged)
                if tagged:
                    flow = start + offset - birth
                    cost += flow
                    squared += flow * flow
                    evaluations += 1
                    tagged_remaining -= 1
                else:
                    background_admissions += 1

    admit(root, Fraction(0))
    while tagged_remaining:
        lane = min(range(len(lane_ends)), key=lambda index: (lane_ends[index], index))
        start = lane_ends[lane]
        while cursor < len(events) and events[cursor][0] <= start:
            birth, coordinate = events[cursor]
            waiting[coordinate].append((birth, True))
            population.append(coordinate, birth, True, cursor)
            counts[coordinate] += 1
            dispatch_work["observed_arrival_counter_updates"] += 1
            cursor += 1
        if not suffix.release_through(start):
            overflow = True
            break
        while background_cursor < len(suffix.events):
            birth, coordinate = suffix.events[background_cursor]
            if birth > start:
                break
            waiting[coordinate].append((birth, False))
            population.append(coordinate, birth, False, background_cursor)
            counts[coordinate] += 1
            dispatch_work["observed_arrival_counter_updates"] += 1
            background_cursor += 1
        current_queue = tuple(len(group) for group in waiting)
        current_ages = tuple(tuple(float(start - birth) for birth, _ in group) for group in waiting)
        current = (planner.options, current_queue, tuple(float(origin + end) for end in lane_ends),
            float(origin + start), lane, current_ages, tuple(counts))
        dispatch_work["age_context_job_reads"] += sum(current_queue)
        dispatch_work["observed_arrival_context_coordinate_reads"] += len(counts)
        dispatch_work["lane_context_reads"] += len(lane_ends)
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
            continue
        admit(chosen, start)
        lane_ends[lane] = start + planner.durations[chosen]

    final_tree = capture_planner_tree(planner, restore_work)
    if planner.decision_counter != after_tree["planners"][0]["decision_counter"] + calls:
        raise ValueError("free finite controller did not consume every dispatch")
    return dict(cost=cost, squared=squared, calls=calls, idle_calls=idle_calls,
        evaluations=evaluations, admitted_reads=admitted_reads, tag_reads=tag_reads,
        background_admissions=background_admissions, overflow=overflow,
        complete=not overflow and not tagged_remaining, tagged_remaining=tagged_remaining,
        population=population.result(cost, squared), **_background_result(suffix),
        logical_work=dict(logical_work), proposal_status=dict(statuses), dispatch_work=dict(dispatch_work),
        restore_work=dict(restore_work), initial_tree=initial_tree, first_call_context=first_call_context,
        first_native_decision=first_native_decision, final_tree=final_tree,
        actual_work=planner.reuse_work, worker_cpu_s=process_time() - cpu,
        worker_wall_s=perf_counter() - wall)


def evaluate_root_sample(options, planning_seed, paths, before_tree, after_tree, context,
        root, sample_index, *, sample_count=8, event_cap=128, workers=6):
    """Evaluate one root/global-sample unit; retain all exact partial overflow data.

    ``context`` has queue/available/now/lane/ages/counts. ``paths`` are the
    original complete outer samples. Both trees belong to the finite planner,
    immediately before/after consuming its complete native anchor proposal.
    The approximate branch receives the native post-prefix RNG at the anchor
    counter and before-child states. The free branch receives the full after
    tree. Background births use the same fixed anchor seed/rates in both arms.
    Root means are usable only when all eight paired units complete.
    """
    if len(paths) != sample_count or not 0 <= sample_index < sample_count:
        raise ValueError("the original full sample roster and global index are required")
    if len(before_tree["planners"]) != 3 or len(after_tree["planners"]) != 3:
        raise ValueError("finite before/after trees must contain all three controllers")
    anchor = before_tree["planners"][0]["decision_counter"]
    if after_tree["planners"][0]["decision_counter"] != anchor + 1:
        raise ValueError("after tree must have consumed exactly one complete native proposal")
    path = paths[sample_index]
    if isinstance(path, PlanningArrivals):
        if path.overflow:
            raise ValueError("overflowing original prefix population cannot be scored")
        path = path.events
    events = tuple((_exact(event[0]), event[1]) for event in path)
    queue = tuple(context["queue"])
    ages = tuple(tuple(group) for group in context["ages"])
    counts = tuple(context["counts"])
    if len(ages) != len(queue) or any(len(group) != count for group, count in zip(ages, queue, strict=True)):
        raise ValueError("visible FIFO population differs from the supplied queue")
    rates = tuple(count / context["now"] if context["now"] > 0 else 0. for count in counts)
    after = after_tree["planners"][0]
    if tuple(after["arrival_rates"]) != rates or tuple(after["observed_counts"]) != counts:
        raise ValueError("after proposal rates/counts differ from the anchor context")
    payload = dict(queue=queue, available=tuple(context["available"]), now=context["now"],
        lane=context["lane"], ages=ages, outer_state=(after["rng_state"], anchor, rates, counts),
        input_states=tuple(_tree_semantic(saved) for saved in before_tree["planners"][1:]),
        root=root, sample_start=sample_index, paths=(path,))
    approx = _score_approx(tuple(options), planning_seed, sample_count, event_cap,
        payload, _Population(queue, ages, events))
    constructor_cpu, constructor_wall = process_time(), perf_counter()
    with native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
            options, planning_seed, sample_count, event_cap, workers=workers) as planner:
        constructor_cpu, constructor_wall = process_time() - constructor_cpu, perf_counter() - constructor_wall
        if any(need > count for need, count in zip(planner.requirements[root], queue, strict=True)):
            raise ValueError("registered physical root is infeasible")
        seed = f"{planning_seed}:background:{anchor}:{sample_index}"
        suffix = _LazyBackgroundArrivals(rates, planner.maximum_duration, seed, event_cap, len(events))
        free = _score_free(planner, context, root, events, suffix, after_tree, _Population(queue, ages, events))
        free.update(constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
            preprocessing=dict(planner.preprocessing_work), controller_constructors=3)
        environment = dict(rates=rates, window_s=planner.maximum_duration, seed=seed,
            event_cap=event_cap, sample_index=sample_index, anchor_counter=anchor, prefix_events=events)
    common = min(len(approx["background_events"]), len(free["background_events"]))
    if approx["background_events"][:common] != free["background_events"][:common]:
        raise ValueError("paired synthetic background prefixes differ")
    if approx["population"]["tagged_population"] != free["population"]["tagged_population"]:
        raise ValueError("paired tagged populations differ")
    return dict(root=root, sample_index=sample_index, environment=environment,
        tagged_population=approx["population"]["tagged_population"], approx=approx, free=free,
        paired_complete=approx["complete"] and free["complete"])
