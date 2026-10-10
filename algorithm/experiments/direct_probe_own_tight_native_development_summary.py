"""Direct root completion before native branching, with genuine paid prefixes."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    _affine_work, _canonical_work, audit_affine_stage,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.axis_capacity_frontier_development_summary import audit_axis_capacity_proof
from algorithm.experiments.affine_dual_fresh_confirmation_summary import audit_retained_completion
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work
from algorithm.experiments.early_root_canonical_development_summary import LEGACY_FIELDS, _catalogue, _group as _control_group
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.fused_root_dual_face_development_summary import (
    DUAL_FACE_COUNTERS, ROOT_SETTLE_COUNTERS as OLD_ROOT_SETTLE_COUNTERS, ROOT_SETTLE_FEE_COUNTERS, ROUTING_COUNTERS,
    _audit_fused_root_result, _work as _fused_work,
)
from algorithm.experiments.own_tight_dual_native_development_summary import (
    _audit_own_tight_result as _parent_own_result, _work as _parent_own_work,
)
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _exact, _metric
from algorithm.experiments.root_pruning_development_summary import audit_root_stage, _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.settled_root_dual_face_checked_development_summary import (
    LP_FIELDS, STATE_FIELDS, _score, audit_settled_dual_face,
)
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same
from algorithm.experiments.zero_delta_affine_development_summary import ZERO_COUNTERS, audit_zero_stage

OWN_TIGHT_COUNTERS = (
    "checks", "bound_comparisons", "certificate_reads", "own_bound_reads", "own_bound_comparisons",
    "variable_visits", "lower_flag_reads", "upper_price_reads", "upper_price_comparisons",
    "box_coordinate_reads", "box_coordinate_comparisons", "bound_values_copied", "bound_values_written",
    "own_bound_cache_reads", "own_bound_cache_writes", "lower_feasibility_rechecks", "capacity_rechecks", "threshold_rechecks",
    "canonical_rechecks", "event_records_written", "event_group_reads", "event_requirement_reads",
    "event_price_reads", "event_bound_values_copied", "event_exact_position_reads", "event_fraction_pairs_written",
    "event_rational_operations", "event_snapshot_entries_copied", "tight_regions", "changed_regions",
    "changed_lower_coordinates", "changed_upper_coordinates", "empty_regions",
)
OWN_TIGHT_FEE_COUNTERS = OWN_TIGHT_COUNTERS[:-5]
OWN_FIELDS = ("native_own_tight_work", "native_own_tight_evidence")
ROOT_SETTLE_COUNTERS = (*OLD_ROOT_SETTLE_COUNTERS, "stopped_for_branch")


def _without_own(stage):
    return {name: value for name, value in stage.items() if name not in OWN_FIELDS}


def _native_math(stage):
    return {**stage, OWN_FIELDS[0]: {name: value for name, value in stage[OWN_FIELDS[0]].items()
                                   if name != "event_snapshot_entries_copied"}}


def _event_errors(event, model, queue, lanes):
    """Rebuild source/current dual objectives and simultaneous endpoint fixes."""
    errors = []
    options, dimension = model["options"], len(queue)
    grouped = {}
    durations = [Fraction.from_float(max(1., float(option["duration"]))) for option in options]
    for index, option in enumerate(options):
        grouped.setdefault(tuple(option["requirements"]), []).append(index)
    requirements = list(grouped)
    minima = [min(durations[index] for index in grouped[required]) for required in requirements]
    threshold = Fraction(*event["threshold_fraction"])
    representatives = {group: next(index for index in grouped[required] if durations[index] <= threshold)
                       for group, required in enumerate(requirements) if minima[group] <= threshold}
    eligible = sorted(representatives, key=representatives.get)
    local = [requirements[group] for group in eligible]
    exact = [position for position, group in enumerate(eligible) if minima[group] == threshold]
    if (threshold not in minima or not _same(event["eligible"], eligible)
            or not _same(event["requirements"], local) or not _same(event["exact_duration_positions"], exact)):
        return ["own-tight event groups and exact-duration positions differ from the original catalogue"]
    counts, indices = event["incumbent_counts"], event["incumbent_indices"]
    score, duration = Fraction(*event["incumbent_score_fraction"]), Fraction(*event["incumbent_duration_fraction"])
    service = [sum(count * option["requirements"][coordinate] for count, option in zip(counts, options, strict=True))
               for coordinate in range(dimension)]
    expanded = [index for index, count in enumerate(counts) for _ in range(count)]
    if (any(type(count) is not int or count < 0 for count in counts) or sum(counts) > lanes
            or any(done > waiting for done, waiting in zip(service, queue, strict=True))
            or not _same(indices, expanded + [None] * (lanes - len(expanded)))
            or duration != max([Fraction(1), *(value for count, value in zip(counts, durations, strict=True) if count)])
            or score != Fraction(sum(waiting * done for waiting, done in zip(queue, service, strict=True))) / duration
            or Fraction(*event["bound_fraction"]) != score):
        errors.append("own-tight event incumbent must be its feasible original action and exact bound trigger")
    prices = [Fraction(*pair) for pair in event["certificate_prices"]]
    size = len(local)
    if len(prices) != 1 + dimension + size or any(price < 0 for price in prices):
        return errors + ["own-tight event requires its complete nonnegative original LP dual prices"]
    weights = [sum(waiting * amount for waiting, amount in zip(queue, required, strict=True)) for required in local]
    upper_prices = prices[1 + dimension:]
    alpha = [weight - prices[0] - sum(prices[1 + coordinate] * amount for coordinate, amount in enumerate(required))
             - cap for weight, required, cap in zip(weights, local, upper_prices, strict=True)]
    if any(value > 0 for value in alpha):
        errors.append("own-tight source dual must be exactly column feasible")
    capacity = [min(lanes, *(waiting // amount for waiting, amount in zip(queue, required, strict=True) if amount)) for required in local]
    source_l, source_u = event["certificate_lower"], event["certificate_upper"]
    anchor_l, anchor_u = event["anchor_lower"], event["anchor_upper"]
    before_l, before_u = event["before_lower"], event["before_upper"]
    if any(len(values) != size for values in (source_l, source_u, anchor_l, anchor_u, before_l, before_u)):
        return errors + ["own-tight source, cached anchor and current boxes must retain every eligible coordinate"]
    if (any(type(value) is not int for values in (source_l, source_u, anchor_l, anchor_u, before_l, before_u) for value in values)
            or any(not 0 <= sl <= al <= bl <= bu <= au <= su <= cap
                   for sl, al, bl, bu, au, su, cap in zip(source_l, anchor_l, before_l, before_u, anchor_u, source_u, capacity, strict=True))
            or sum(before_l) > lanes or any(sum(count * required[c] for count, required in zip(before_l, local, strict=True)) > waiting
                                          for c, waiting in enumerate(queue))):
        errors.append("own-tight current and cached boxes must descend from the physical certificate anchor")
    source = sum(count * weight for count, weight in zip(source_l, weights, strict=True))
    source += prices[0] * (lanes - sum(source_l))
    source += sum(prices[1 + coordinate] * (queue[coordinate] - sum(count * required[coordinate]
                  for count, required in zip(source_l, local, strict=True))) for coordinate in range(dimension))
    source += sum(cap * (upper - lower) for cap, lower, upper in zip(upper_prices, source_l, source_u, strict=True))

    def own(lower, upper):
        return source + sum((left - original) * coefficient for left, original, coefficient in zip(lower, source_l, alpha, strict=True)) + sum(
            (right - original) * cap for right, original, cap in zip(upper, source_u, upper_prices, strict=True))

    target, current = score * threshold, own(before_l, before_u)
    if (event["anchor_kind"] not in ("certificate", "affine")
            or event["anchor_kind"] == "certificate" and (not _same(anchor_l, source_l) or not _same(anchor_u, source_u))
            or Fraction(*event["certificate_numerator_fraction"]) != source
            or Fraction(*event["anchor_numerator_fraction"]) != own(anchor_l, anchor_u)
            or Fraction(*event["current_own_numerator_fraction"]) != current
            or Fraction(*event["target_numerator_fraction"]) != target
            or Fraction(*event["own_bound_fraction"]) != current / threshold or current != target):
        errors.append("own-tight fixing must use the independently recomputed current own objective, never an inherited minimum")
    face_l, face_u = list(before_l), list(before_u)
    for position, (left, right, coefficient, cap) in enumerate(zip(before_l, before_u, alpha, upper_prices, strict=True)):
        if left < right:
            if coefficient < 0:
                face_u[position] = left
            if cap > 0:
                face_l[position] = right
    if (not _same(event["face_lower"], face_l) or not _same(event["face_upper"], face_u)
            or face_l == list(before_l) and face_u == list(before_u)):
        errors.append("own-tight evidence must record a real simultaneous negative-lower/positive-upper endpoint restriction")
    after_u = list(face_u)
    direct_empty = any(left > right for left, right in zip(face_l, face_u, strict=True))
    residual = [waiting - sum(count * required[c] for count, required in zip(face_l, local, strict=True))
                for c, waiting in enumerate(queue)]
    infeasible = sum(face_l) > lanes or min(residual, default=0) < 0
    if not direct_empty and not infeasible and face_l != list(before_l):
        for position, required in enumerate(local):
            after_u[position] = min(face_u[position], face_l[position] + lanes - sum(face_l),
                *(face_l[position] + residual[c] // amount for c, amount in enumerate(required) if amount))
    if not _same(event["after_lower"], face_l) or not _same(event["after_upper"], after_u):
        errors.append("own-tight post-face bounds must retain their real mandatory-lower capacity propagation")
    if direct_empty:
        disposition = "direct_empty"
    elif infeasible:
        disposition = "infeasible_lower"
    elif not any(after_u[position] >= 1 for position in exact):
        disposition = "threshold_empty"
    else:
        optimistic = list(face_l)
        remaining = lanes - sum(optimistic)
        for position in range(size):
            copies = min(after_u[position] - optimistic[position], remaining)
            optimistic[position] += copies
            remaining -= copies
        encoded = [representatives[group] for position, group in enumerate(eligible) for _ in range(optimistic[position])]
        face_key = threshold, tuple(encoded + [len(options)] * (lanes - len(encoded)))
        incumbent_key = duration, tuple(len(options) if index is None else index for index in indices)
        disposition = "canonical_pruned" if face_key >= incumbent_key else "kept"
        if own(face_l, after_u) != target:
            errors.append("nonempty endpoint fixing and capacity propagation must preserve the exact cached own objective")
    if event["disposition"] != disposition:
        errors.append("own-tight close/keep reason differs from its actual face, physical capacity, duration or canonical key")
    return errors


def _own_errors(stage, model, queue, lanes, previous=None):
    work, events = stage.get(OWN_FIELDS[0], {}), stage.get(OWN_FIELDS[1], ())
    if set(work) != set(OWN_TIGHT_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return ["native own-tight work must retain all33 actual nonnegative integer counters"]
    errors = []
    for event in events:
        errors.extend(_event_errors(event, model, queue, lanes))
        if Fraction(*event["incumbent_score_fraction"]) > _score(stage["selection"], queue):
            errors.append("own-tight event incumbent cannot exceed the native checkpoint incumbent")
    size = sum(len(event["eligible"]) for event in events)
    changed_l = sum(sum(left != old for left, old in zip(event["face_lower"], event["before_lower"], strict=True)) for event in events)
    changed_u = sum(sum(right != old for right, old in zip(event["face_upper"], event["before_upper"], strict=True)) for event in events)
    direct = sum(event["disposition"] == "direct_empty" for event in events)
    infeasible = sum(event["disposition"] == "infeasible_lower" for event in events)
    threshold = sum(event["disposition"] == "threshold_empty" for event in events)
    lower_checks = sum(event["disposition"] != "direct_empty" and event["face_lower"] != event["before_lower"] for event in events)
    affine = sum(event["anchor_kind"] == "affine" for event in events)
    visits = work["variable_visits"]
    expected = dict(bound_comparisons=work["checks"], own_bound_comparisons=work["own_bound_reads"],
        lower_flag_reads=visits, upper_price_reads=visits, upper_price_comparisons=visits,
        box_coordinate_reads=2 * visits, box_coordinate_comparisons=visits, bound_values_copied=4 * visits,
        bound_values_written=changed_l + changed_u, lower_feasibility_rechecks=lower_checks,
        capacity_rechecks=lower_checks - infeasible, threshold_rechecks=len(events) - direct - infeasible,
        canonical_rechecks=len(events) - direct - infeasible - threshold, event_records_written=len(events),
        event_group_reads=2 * size, event_requirement_reads=size * len(queue),
        event_price_reads=sum(len(event["certificate_prices"]) for event in events),
        event_bound_values_copied=4 * size + 2 * sum(len(event["eligible"]) for event in events if event["anchor_kind"] == "affine"),
        event_exact_position_reads=sum(len(event["exact_duration_positions"]) for event in events),
        event_fraction_pairs_written=sum(len(event["certificate_prices"]) + 9 for event in events),
        event_rational_operations=2 * len(events) + affine, changed_regions=len(events),
        changed_lower_coordinates=changed_l, changed_upper_coordinates=changed_u, empty_regions=direct + infeasible + threshold)
    different = [name for name, value in expected.items() if work[name] != value]
    if different:
        errors.append("own-tight fees or changed-domain diagnostics differ from their actual evidence: " + ", ".join(different))
    groups, _ = _catalogue(model)
    tight = work["tight_regions"]
    replacements = work["own_bound_cache_reads"] - stage["selection"]["lp_calls"] - stage["selection"]["branch_splits"] - len(events)
    if (not work["checks"] >= work["certificate_reads"] >= work["own_bound_reads"] >= tight >= len(events)
            or not max(tight, size) <= visits <= groups * tight
            or not 0 <= replacements <= min(stage["capacity_propagation"]["region_visits"], stage["capacity_propagation"]["tightened_bounds"])
            or work["own_bound_cache_writes"] - work["own_bound_cache_reads"] != stage["selection"]["branch_splits"] + stage["dual_reuse"]["child_bound_evaluations"]):
        errors.append("native own-tight gates, support visits or cached own-value assignments differ from genuine native work")
    if previous is not None and (any(work[name] < previous[OWN_FIELDS[0]][name] for name in OWN_TIGHT_COUNTERS)
            or not _same(events[:len(previous[OWN_FIELDS[1]])], previous[OWN_FIELDS[1]])):
        errors.append("native own-tight resumption decreased cumulative paid work or changed earlier immutable events")
    return errors


def _phase_errors(result, state):
    root, settled, work = (result[name] for name in ("root_lp", "settled_lp", "root_settle_work"))
    triggered = root["selection"]["score_exact"] and not root["selection"]["canonical_exact"]
    if not triggered:
        return [] if settled is None and work is None else ["only a genuine raw score-certified canonical-open root may settle"]
    if settled is None or work is None:
        return ["every raw score-certified canonical-open root must retain its actual native settlement"]
    if set(work) != set(ROOT_SETTLE_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return ["direct-probe settlement must retain all16 actual phase counters"]
    errors = []
    before, after = root["selection"], settled["selection"]
    if (work["calls"] != 1 or work["snapshot_calls"] != 1
            or work["lp_calls_before"] != before["lp_calls"] or work["lp_calls_after"] != after["lp_calls"]
            or any(after[name] != before[name] for name in LP_FIELDS)
            or after["lp_node_budget"] != result["root_lp_budget"]
            or not after["score_exact"] or _score(after, state["queue"]) != _score(before, state["queue"])):
        errors.append("own-tight root settlement must use zero extra LP and preserve the exact M incumbent score")
    if (work["frontier_regions_before"] != result["routing_preparation"]["root_snapshot_heap_entries_read"]
            or work["snapshot_heap_entries_read"] != work["frontier_regions_after"]
            or work["snapshot_retained_entries_read"] != 0
            or after["frontier_regions_remaining"] > work["frontier_regions_after"]):
        errors.append("own-tight settlement must pay one actual post-settle raw heap scan with no retained-list scan")
    actual_prunes = after["bound_pruned_regions"] - (work["frontier_regions_after"] - after["frontier_regions_remaining"])
    actual_prunes -= before["bound_pruned_regions"] - (work["frontier_regions_before"] - before["frontier_regions_remaining"])
    own_empty = settled[OWN_FIELDS[0]]["empty_regions"] - root[OWN_FIELDS[0]]["empty_regions"]
    stops = work["stopped_for_lp"] + work["stopped_for_branch"]
    if (work["split_regions"] != 0
            or work["popped_regions"] != work["pruned_regions"] + work["singleton_regions"] + work["stopped_for_branch"]
            or work["head_checks"] != work["popped_regions"] + work["stopped_for_lp"]
            or stops != int(work["frontier_regions_after"] > 0)
            or not work["stopped_for_lp"] <= work["unsolved_prune_checks"] <= work["pruned_regions"] + work["stopped_for_lp"]
            or work["frontier_regions_after"] != work["frontier_regions_before"] - work["pruned_regions"] - work["singleton_regions"]
            or work["singleton_regions"] + work["stopped_for_branch"] > root["root_pruning"]["roots_solved"]
            or after["branch_splits"] != before["branch_splits"]
            or after["singleton_regions_closed"] - before["singleton_regions_closed"] != work["singleton_regions"]
            or actual_prunes < 0 or actual_prunes + own_empty < work["pruned_regions"]
            or after["canonical_exact"] != (stops == 0)):
        errors.append("direct-probe phase must preserve its complete clipped branch box without splitting, while accounting for actual closures and lazy snapshot pruning")
    return errors


def _snapshot_errors(result, state):
    """Charge separately constructed checkpoints, excluding aliased fields."""
    errors = []
    native = state["candidate_native_root"]
    audit = [native["root_lp"]]
    if native["settled_lp"] is not None:
        audit.append(native["settled_lp"])
    audit.extend(state["candidate_native_traces"][str(cap)] for cap in state["caps"][1:])
    whole = [result["root_lp"]]
    if result["settled_lp"] is not None:
        whole.append(result["settled_lp"])
    roots = result["root_lp_budget"]
    if result["initial_lp"]["selection"]["lp_node_budget"] != roots:
        whole.append(result["initial_lp"])
    if result["resumed"]:
        whole.append(result["final_lp"])
    for label, stages in (("native audit", audit), ("whole selector", whole)):
        copies = 0
        for stage in stages:
            copies += len(stage[OWN_FIELDS[1]])
            if stage[OWN_FIELDS[0]]["event_snapshot_entries_copied"] != copies:
                errors.append(label + " checkpoint must charge every actually materialized cumulative event tuple once")
        if label == "whole selector" and result["final_lp"][OWN_FIELDS[0]]["event_snapshot_entries_copied"] != copies:
            errors.append("aliased final native snapshot must retain its genuine last constructed checkpoint copy fee")
    return errors


def _work(results, label):
    if label == "own":
        return _parent_own_work(results, "candidate")
    work = _fused_work(results, "candidate" if label == "fused" else label)
    if label == "candidate":
        stages = [result["final_lp"] for result in results]
        work["native_own_tight_work"] = {name: sum(stage[OWN_FIELDS[0]][name] for stage in stages)
            for name in OWN_TIGHT_FEE_COUNTERS}
        work["native_own_tight_diagnostics"] = {name: sum(stage[OWN_FIELDS[0]][name] for stage in stages)
            for name in OWN_TIGHT_COUNTERS if name not in OWN_TIGHT_FEE_COUNTERS}
        phases = [result["root_settle_work"] for result in results if result["root_settle_work"] is not None]
        work["root_settle_diagnostics"] = {name: sum(phase[name] for phase in phases)
            for name in ROOT_SETTLE_COUNTERS if name not in ROOT_SETTLE_FEE_COUNTERS}
    return work


def _kernel_work(stages):
    return {**_native_work(stages), "dual_reuse": _dual_work(stages), "affine_dual": _affine_work(stages),
        "zero_delta": {name: sum(stage["zero_delta"][name] for stage in stages) for name in ZERO_COUNTERS},
        "canonical_work": _canonical_work([stage["selection"] for stage in stages]),
        "native_own_tight_work": {name: sum(stage[OWN_FIELDS[0]][name] for stage in stages)
            for name in OWN_TIGHT_FEE_COUNTERS},
        "native_own_tight_diagnostics": {name: sum(stage[OWN_FIELDS[0]][name] for stage in stages)
            for name in OWN_TIGHT_COUNTERS if name not in OWN_TIGHT_FEE_COUNTERS}}


def _stage_errors(stage, previous, model, queue, lanes, cap, expected, state):
    selected, roots = stage["selection"], len(stage["thresholds"])
    errors = audit_root_stage(stage)
    if (selected["lp_node_budget"] != cap
            or selected["lp_calls"] + roots - stage["root_pruning"]["roots_solved"] > cap):
        errors.append("native prefix differs from its actual cap or reserved root slots")
    metric = _metric(selected, state, model, lanes, cap, expected, stage["preprocessing"])
    if metric["failures"]:
        errors.append("native prefix has invalid physical action, certificate or actual cumulative LP work")
    errors.extend(audit_affine_stage(stage, len(queue), previous))
    errors.extend(audit_zero_stage(stage, previous))
    errors.extend(_own_errors(stage, model, queue, lanes, previous))
    if previous is not None:
        if (not _same(stage["root_pruning"], previous["root_pruning"])
                or not _same(stage["thresholds"], previous["thresholds"])
                or any(stage["preprocessing"][name] != value for name, value in previous["preprocessing"].items()
                       if name != "lift_option_probes")
                or any(stage["capacity_propagation"][name] < value for name, value in previous["capacity_propagation"].items())
                or any(selected[name] < previous["selection"][name] for name in COUNTERS if name != "bound_pruned_regions")
                or _score(selected, queue) < _score(previous["selection"], queue)):
            errors.append("native resumption repeated roots/preprocessing or decreased actual cumulative work or incumbent score")
    return errors


def _prefix_errors(result, state, protocol, model, lanes, expected):
    errors = []
    native_root, traces = state["candidate_native_root"], state["candidate_native_traces"]
    root, settled, phase = (result[name] for name in ("root_lp", "settled_lp", "root_settle_work"))
    groups, thresholds = _catalogue(model)
    roots = len(thresholds)
    caps = (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"])
    if tuple(state["caps"]) != caps or set(traces) != {str(cap) for cap in caps}:
        return ["the own native engine must preserve all four registered actual checkpoints"]
    if (set(native_root) != {"root_lp", "settled_lp", "root_settle_work"}
            or any(not _same(result[name], native_root[name]) for name in native_root)
            or not _same(traces[str(roots)], settled if settled is not None else root)):
        errors.append("whole root/settled/phase records differ from the separately constructed actual own native engine")
    parent = state["previous_own_tight_routed"]
    if (not _same(root, parent["root_lp"]) or result["root_lp_budget"] != roots
            or not _same(root["thresholds"], thresholds) or root["preprocessing"]["requirement_group_count"] != groups):
        errors.append("the pre-elimination M checkpoint must retain every genuine frozen fused native field")
    errors.extend("raw root: " + message for message in audit_reserved_stage(root, roots, roots))
    errors.extend("raw root: " + message for message in _stage_errors(root, None, model, state["queue"], lanes, roots, expected, state))
    raw_work = root[OWN_FIELDS[0]]
    if root[OWN_FIELDS[1]] or any(raw_work[name] for name in OWN_TIGHT_COUNTERS if name not in
            ("own_bound_cache_reads", "own_bound_cache_writes", "event_snapshot_entries_copied")):
        errors.append("raw M must be frozen before any own-tight check, shrink or event")
    preparation = result["routing_preparation"]
    if (set(preparation) != set(ROUTING_COUNTERS)
            or any(type(value) is not int or value < 0 for value in preparation.values())
            or any(preparation[name] for name in ROUTING_COUNTERS[:8])
            or preparation["root_snapshot_calls"] != 1 or preparation["root_snapshot_retained_entries_read"] != 0
            or preparation["root_snapshot_heap_entries_read"] != parent["routing_preparation"]["root_snapshot_heap_entries_read"]):
        errors.append("root preparation must pay only its genuine one raw heap pass")
    errors.extend(_phase_errors(result, state))
    previous = root
    if settled is not None:
        errors.extend("settled: " + message for message in _stage_errors(settled, previous, model, state["queue"], lanes, roots, expected, state))
        previous = settled
    for cap in caps[1:]:
        stage = traces[str(cap)]
        errors.extend(f"native cap{cap}: " + message for message in _stage_errors(
            stage, previous, model, state["queue"], lanes, cap, expected, state))
        previous = stage
    for name in ("initial_lp", "final_lp"):
        stage = result[name]
        cap = stage["selection"]["lp_node_budget"]
        if cap not in (roots, protocol["initial_lp_budget"], protocol["lp_node_budget"]) or not _same(_native_math(stage), _native_math(traces[str(cap)])):
            errors.append(name + " differs from its actual same-search own native checkpoint")
    errors.extend(_snapshot_errors(result, state))
    return errors


def _audit_direct_probe_result(result, state, protocol, model, lanes, label, expected):
    if label == "own":
        parent_state = {**state, "candidate_native_root": state["previous_own_tight_native_root"],
                       "candidate_native_traces": state["previous_own_tight_native_traces"]}
        metric, errors, quality = _parent_own_result(result, parent_state, protocol, model, lanes, "candidate", expected)
        if not _same(result, state["previous_own_tight_routed"]):
            errors.append("own control differs from its actual frozen whole-selector result")
        return metric, errors, quality
    if label != "candidate":
        metric, errors, quality = _audit_fused_root_result(result, state, protocol, model, lanes,
            "candidate" if label == "fused" else label, expected)
        if label == "fused" and not _same(result, state["previous_fused_root_routed"]):
            errors.append("fused control differs from its actual frozen whole-selector result")
        return metric, errors, quality
    errors = _prefix_errors(result, state, protocol, model, lanes, expected)
    for name in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[name] != protocol[name]:
            errors.append("own-tight " + name + " differs from registration")
    root, settled, initial, final, probe = (result[name] for name in ("root_lp", "settled_lp", "initial_lp", "final_lp", "probe"))
    roots = result["root_lp_budget"]
    if root["selection"]["canonical_exact"]:
        if result["dual_face_evidence"] is not None or any(result["dual_face_work"].values()):
            errors.append("complete raw roots cannot export or charge an outer dual face")
        if (result["status"] != "LP_ROOT_COMPLETE" or result["probe_stage"] != "none" or probe is not None
                or result["resumed"] or result["frontier_evidence"] is not None
                or not _same(initial, root) or not _same(final, root) or not _same(result["selection"], root["selection"])):
            errors.append("complete raw roots must return their own action without an extra probe or continuation")
    elif settled is not None:
        errors.extend(audit_settled_dual_face(result, model, state["queue"], lanes))
        if result["frontier_evidence"] is not None:
            errors.append("score-certified root completion cannot add a late score-proof frontier")
        if settled["selection"]["canonical_exact"]:
            if (probe is not None or result["probe_stage"] != "none" or result["resumed"]
                    or result["status"] != "LP_ROOT_SETTLE_COMPLETE" or not _same(initial, settled)
                    or not _same(final, settled) or not _same(result["selection"], settled["selection"])):
                errors.append("closed own-tight settlement must return its actual M prefix without an integer probe")
        elif probe is None or result["probe_stage"] != "root":
            errors.append("open own-tight settlement must retain its unique paid root probe")
        else:
            errors.extend(_probe_errors(result, state, protocol, model, lanes, expected, settled, roots))
            if probe["selection"]["canonical_exact"]:
                if (result["status"] != "INTEGER_PROBE_COMPLETE" or result["resumed"]
                        or not _same(initial, settled) or not _same(final, settled)
                        or not _same(result["selection"], probe["selection"])):
                    errors.append("completed own-tight root probe must return its real action at M")
            else:
                should_resume = not initial["selection"]["canonical_exact"]
                status = ("LP_RESUME_COMPLETE" if final["selection"]["canonical_exact"] else "LP_RESUME_INCOMPLETE") if should_resume else "LP_INITIAL_COMPLETE"
                if (probe["completion"]["status"] != "STATE_BUDGET_EXHAUSTED"
                        or not _same(probe["selection"], settled["selection"])
                        or probe["score_proof"]["states_visited"] + probe["completion"]["states_visited"] != protocol["integer_state_budget"]
                        or initial["selection"]["lp_node_budget"] != protocol["initial_lp_budget"]
                        or result["resumed"] != should_resume or result["status"] != status
                        or final["selection"]["lp_node_budget"] != (protocol["lp_node_budget"] if should_resume else protocol["initial_lp_budget"])
                        or not _same(result["selection"], final["selection"])):
                    errors.append("exhausted own-tight root completion must retain all budget and continue only the actual native144/256 path")
    else:
        legacy = {name: result[name] for name in LEGACY_FIELDS}
        # The ordinary router compares mathematical prefixes. Their separately
        # paid event-tuple checkpoints are audited in _snapshot_errors above.
        legacy = {**legacy, "initial_lp": _native_math(initial), "final_lp": _native_math(final)}
        current_state = {**state, "traces": {**state["traces"], "new": {
            cap: _native_math(stage) for cap, stage in state["candidate_native_traces"].items()}}}
        _, extra, _ = audit_frontier_confirmation_router(legacy, current_state, protocol, model, lanes, "new", expected)
        errors.extend(extra)
        if result["dual_face_evidence"] is not None or any(result["dual_face_work"].values()):
            errors.append("untriggered root face completion must charge no outer dual-face work")
        if result["probe_stage"] == "frontier_late" and probe is not None and result["frontier_evidence"] is not None:
            errors.extend(audit_axis_capacity_proof(probe, result["frontier_evidence"], state["queue"], lanes))
            errors.extend(audit_retained_completion(probe, model, state["queue"], lanes))
    budget = final["selection"]["lp_node_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if result["selection"]["lp_node_budget"] != budget or metric["failures"]:
        errors.append("returned own-tight action has invalid feasibility, canonical certificate or cumulative LP accounting")
    return metric, errors, _exact(result["selection"], metric, expected)


def _probe_errors(result, state, protocol, model, lanes, expected, baseline, cap):
    probe, errors = result["probe"], []
    try:
        _, extra, _ = _audit_strict(probe, state, {**protocol, "lp_budgets": {"candidate": cap}}, model, lanes, "candidate", expected)
        errors.extend("own-tight root probe: " + message for message in extra)
    except ValueError as error:
        errors.append("own-tight root probe: " + str(error))
    if any(not _same(probe[field], baseline[source]) for field, source in (
            ("baseline_selection", "selection"), ("preprocessing", "preprocessing"), ("capacity_propagation", "capacity_propagation"))):
        errors.append("root probe must preserve its actual own-tight settled baseline and paid native work")
    return errors


def _group(states):
    result = _control_group(states)
    result["executed"]["whole_selector"] = _work([state["routed"] for state in states], "candidate")
    result["executed"]["native"] = _kernel_work([state["candidate_native_traces"][str(state["caps"][-1])] for state in states])
    for label, field in (("own", "previous_own_tight_routed"), ("fused", "previous_fused_root_routed"), ("settled", "previous_settled_root_routed"),
            ("native", "previous_native_root_routed"), ("face", "previous_dual_face_routed"), ("early", "previous_early_routed")):
        result["borrowed"]["whole_selector"][label] = _work([state[field] for state in states], label)
    return result


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered own-tight cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, entries, groups = [], [], [], defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if len(states) != len(cell["states"]) or not states or any(not _same(
                {name: state.get(name) for name in original}, original)
                for state, original in zip(states, cell["states"], strict=True)):
            raise ValueError("every frozen queue and its complete source pointers must be retained")
        count, point = len(states), {"node": row["node"], "lanes": row["lanes"]}
        phase_calls = sum(state["routed"]["root_settle_work"] is not None for state in states)
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options) or row["new_reference_calls"] or row["new_oracle_calls"]
                or row["candidate_calls"] != count or row["native_advance_calls"] != 4 * count
                or row["native_search_constructions"] != 2 * count or row["root_settle_calls"] != phase_calls
                or any(row[name]["cpu_affinity"] != [protocol["cpu"]] or not row[name]["gc_enabled"]
                       for name in ("runtime_start", "runtime_end"))):
            failures.append({**point, "errors": ["registered runtime, model or actual whole/native cold-call coverage differs"]})
        for state in states:
            pointer = {**point, **{name: state[name] for name in ("state_id", "queue", "timing_group")}}
            errors, expected = [], None
            reference = selection_from_record(state["reference"])
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            if state["reference"]["lp_node_budget"] != protocol["reference_budget"] or state["reference_metrics"]["failures"]:
                errors.append("borrowed full reference has invalid budget, certificate or paid work")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            caps = {str(cap) for cap in state["caps"]}
            borrowed = (state["traces"]["old"], state["traces"]["new"], state["previous_zero_traces"],
                state["original_affine_traces"], state["previous_fused_root_traces"])
            if any(set(trace) != caps for trace in borrowed):
                raise ValueError("all genuine borrowed native four-cap ladders must remain complete")
            if not _same(state["previous_fused_root_traces"], state["previous_zero_traces"]) or not _same(
                    state["traces"]["new"], state["previous_zero_traces"]):
                errors.append("the parent/zero borrowed native ladder cannot be replaced by the new own-tight ladder")
            for label, trace, has_roots in (("branch", borrowed[0], True), ("zero", borrowed[2], True), ("affine", borrowed[3], True)):
                errors.extend(label + ": " + message for message in _trace_errors(trace, state, options,
                    row["lanes"], expected, reference, has_roots))
            for label, field in (("own", "previous_own_tight_routed"), ("fused", "previous_fused_root_routed"), ("settled", "previous_settled_root_routed"),
                    ("native", "previous_native_root_routed"), ("face", "previous_dual_face_routed"), ("early", "previous_early_routed"),
                    ("zero", "previous_zero_routed"), ("affine", "original_affine_routed"), ("branch", "old_routed")):
                _, extra, _ = _audit_direct_probe_result(state[field], state, protocol, model, row["lanes"], label, expected)
                errors.extend("borrowed " + label + ": " + message for message in extra)
            _, extra, quality = _audit_direct_probe_result(state["routed"], state, protocol, model, row["lanes"], "candidate", expected)
            errors.extend(extra)
            if errors:
                failures.append({**pointer, "errors": errors})
            if not quality:
                nonclosures.append({**pointer, "status": state["routed"]["status"]})
            entries.append((row, state))
            groups[state["timing_group"]].append(state)
    states, count = [state for _, state in entries], len(entries)
    expected_counts = dict(expected_state_count=count, expected_selector_calls=count, expected_native_advance_calls=4 * count,
        expected_native_constructions=2 * count, expected_interface_calls=5 * count, expected_candidate_native_snapshots=4 * count,
        expected_root_settle_calls=sum(state["routed"]["root_settle_work"] is not None for state in states))
    if protocol["expected_group_counts"] != dict(Counter(state["timing_group"] for state in states)) or any(
            protocol[name] != value for name, value in expected_counts.items()):
        failures.append({"errors": ["full frozen cohort coverage or actual interface/construction/phase counts differ"]})
    work = _group(states)
    executed = work["executed"]["whole_selector"]
    execution = not failures
    advance = bool(execution and complete and not nonclosures)
    oracles = [state["oracle"] for state in states if state["oracle"] is not None]
    return {"schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not complete else
            "DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_ADVANCE" if advance else "DIRECT_PROBE_OWN_TIGHT_NATIVE_DEVELOPMENT_HOLD",
        "state_count": count, "independent_confirmation": False, "execution_checks_passed": execution,
        "references_complete": bool(complete), "quality_gate_passed": advance, "work_gate_passed": execution,
        "development_gate_passed": advance, "failures": failures, "nonclosures": nonclosures,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": count,
        "executed_selector_calls": sum(row["candidate_calls"] for row in rows),
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "executed_interface_calls": sum(row["candidate_calls"] + row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_candidate_native_snapshots": sum(len(state["candidate_native_traces"]) for state in states),
        "executed_root_settle_calls": sum(state["routed"]["root_settle_work"] is not None for state in states),
        "executed_routing_preparation_work": executed["routing_preparation"],
        "executed_root_settle_work": executed["root_settle_work"], "root_settle_diagnostics": executed["root_settle_diagnostics"],
        "executed_dual_face_work": executed["dual_face_work"],
        "executed_native_own_tight_work": executed["native_own_tight_work"],
        "executed_native_own_tight_diagnostics": executed["native_own_tight_diagnostics"],
        **work, "groups": {name: _group(items) for name, items in groups.items()},
        "root_probe_states": [{"node": row["node"], "lanes": row["lanes"], **{name: state[name] for name in ("state_id", "queue")},
            "completion_status": state["routed"]["probe"]["completion"]["status"], "work": _work([state["routed"]], "candidate")}
            for row, state in entries if state["routed"]["probe_stage"] == "root"],
        "root_settle_states": [{"node": row["node"], "lanes": row["lanes"], **{name: state[name] for name in ("state_id", "queue")},
            "status": state["routed"]["status"], "work": _work([state["routed"]], "candidate")}
            for row, state in entries if state["routed"]["root_settle_work"] is not None],
        "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in states),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {
            name: sum(oracle["work"][name] for oracle in oracles) for name in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
