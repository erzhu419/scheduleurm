"""Post-outcome audit repair for the unchanged zero-LP root-settling algorithm.

Native snapshots include currently prunable heap entries in their reported
bound-prune count. Remove both checkpoints' lazy entries before comparing the
actual cumulative native pruning with phase diagnostics. The original frozen
audit and its INVALID_EXECUTION result remain preserved.
"""
from __future__ import annotations

from fractions import Fraction

from algorithm.experiments.affine_dual_development_summary import audit_affine_stage
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.dual_face_root_canonical_development_summary import DUAL_FACE_COUNTERS, ROUTING_COUNTERS
from algorithm.experiments.native_root_dual_face_development_summary import (
    _audit_native_root_result, _work as _native_work, summarize as _native_summary,
)
from algorithm.experiments.resumable_probe_development_summary import _exact, _metric
from algorithm.experiments.root_pruning_development_summary import audit_root_stage
from algorithm.experiments.score_face_completion_confirmation_summary import _reference
from algorithm.experiments.strict_score_proof_timing_summary import _audit_result as _audit_strict, _same
from algorithm.experiments.zero_delta_affine_development_summary import audit_zero_stage

ROOT_SETTLE_COUNTERS = (
    "calls", "head_checks", "unsolved_prune_checks", "popped_regions", "pruned_regions", "split_regions",
    "singleton_regions", "stopped_for_lp", "snapshot_calls", "snapshot_heap_entries_read",
    "snapshot_retained_entries_read", "lp_calls_before", "lp_calls_after", "frontier_regions_before", "frontier_regions_after",
)
ROOT_SETTLE_FEE_COUNTERS = (
    "calls", "head_checks", "unsolved_prune_checks", "snapshot_calls", "snapshot_heap_entries_read", "snapshot_retained_entries_read",
)
SETTLE_FIELDS = ("settled_lp", "root_settle_work")
LP_FIELDS = ("lp_calls", "successful_lp_calls", "dual_certified_lp_calls", "safe_fallback_lp_calls", "dual_repair_count",
             "maximum_dual_repair", "lp_matrix_coefficients", "dual_column_checks")
STATE_FIELDS = frozenset(("option_counts", "option_indices", "service", "duration", "score", "upper_bound", "gap", "epsilon",
    "exact", "score_exact", "canonical_exact", "lp_node_budget", "canonical_tie_regions_remaining",
    "frontier_regions_remaining", "inherited_unsolved_regions"))


def _score(selection, queue):
    return Fraction(sum(q * done for q, done in zip(queue, selection["service"], strict=True))) / Fraction.from_float(selection["duration"])


def _without_cap(stage):
    return {**stage, "selection": {name: value for name, value in stage["selection"].items() if name != "lp_node_budget"}}


def _settle_errors(result, state):
    root, settled, work = (result[name] for name in ("root_lp", "settled_lp", "root_settle_work"))
    previous = state["previous_native_root_routed"]
    errors = []
    if (not _same(root, previous["root_lp"]) or result["root_lp_budget"] != previous["root_lp_budget"]
            or not _same(result["routing_preparation"], previous["routing_preparation"])):
        errors.append("raw root M and every actual root snapshot fee must equal the genuine native-root parent")
    traces = state["previous_native_root_traces"]
    if (not _same(traces, state["previous_zero_traces"]) or not _same(traces, state["traces"]["new"])
            or set(traces) != {str(cap) for cap in state["caps"]}):
        errors.append("the real four-cap native-root and zero ladders must remain unchanged")
    triggered = root["selection"]["score_exact"] and not root["selection"]["canonical_exact"]
    if not triggered:
        if settled is not None or work is not None or not _same(
                {name: value for name, value in result.items() if name not in SETTLE_FIELDS}, previous):
            errors.append("untriggered settlement must preserve all native-root fields and charge no phase work")
        return errors
    if settled is None or work is None:
        return errors + ["every score-certified canonical-open raw root requires its actual settled checkpoint and work"]
    if set(work) != set(ROOT_SETTLE_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return errors + ["settlement requires all15 actual nonnegative integer event and diagnostic counters"]
    before, after = root["selection"], settled["selection"]
    if (work["calls"] != 1 or work["snapshot_calls"] != 1
            or work["lp_calls_before"] != before["lp_calls"] or work["lp_calls_after"] != after["lp_calls"]
            or any(after[name] != before[name] for name in LP_FIELDS)
            or after["lp_node_budget"] != result["root_lp_budget"]
            or not after["score_exact"] or _score(after, state["queue"]) != _score(before, state["queue"])):
        errors.append("settlement must retain the actual M cap, LP calls, LP/certificate work and exact incumbent score")
    if (work["frontier_regions_before"] * 2 != result["routing_preparation"]["root_snapshot_heap_entries_read"]
            or work["snapshot_heap_entries_read"] != 2 * work["frontier_regions_after"]
            or work["snapshot_retained_entries_read"] != 2 * after["frontier_regions_remaining"]
            or after["frontier_regions_remaining"] > work["frontier_regions_after"]):
        errors.append("settlement must charge its real raw heap and retained frontier scans separately")
    admitted = work["frontier_regions_after"] - work["frontier_regions_before"] + work["popped_regions"]
    if (work["popped_regions"] != work["pruned_regions"] + work["split_regions"] + work["singleton_regions"]
            or work["head_checks"] != work["popped_regions"] + work["stopped_for_lp"]
            or work["stopped_for_lp"] != int(work["frontier_regions_after"] > 0)
            or not work["stopped_for_lp"] <= work["unsolved_prune_checks"] <= work["pruned_regions"] + work["stopped_for_lp"]
            or not 0 <= admitted <= 2 * work["split_regions"]
            or work["split_regions"] + work["singleton_regions"] > root["root_pruning"]["roots_solved"]
            or after["branch_splits"] - before["branch_splits"] != work["split_regions"]
            or after["singleton_regions_closed"] - before["singleton_regions_closed"] != work["singleton_regions"]
            or (after["bound_pruned_regions"] - (work["frontier_regions_after"] - after["frontier_regions_remaining"])
                - before["bound_pruned_regions"] + (work["frontier_regions_before"] - before["frontier_regions_remaining"])) < work["pruned_regions"]
            or after["canonical_exact"] != (work["stopped_for_lp"] == 0)):
        errors.append("settlement head, free pop/split/singleton and stopping diagnostics differ from native heap progress")
    if (not _same(root["thresholds"], settled["thresholds"]) or not _same(root["root_pruning"], settled["root_pruning"])
            or any(value != settled["preprocessing"][name] for name, value in root["preprocessing"].items() if name != "lift_option_probes")
            or any(after[name] < value for name, value in before.items() if name not in STATE_FIELDS and type(value) in (int, float))
            or any(settled["capacity_propagation"][name] < value for name, value in root["capacity_propagation"].items())):
        errors.append("settlement must retain native preprocessing/roots and actual cumulative search/capacity work")
    return errors


def audit_settled_dual_face(result, model, queue, lanes):
    """Check general child boxes against their original LP certificate anchors."""
    work, evidence, settled = (result.get(name) for name in ("dual_face_work", "dual_face_evidence", "settled_lp"))
    if set(work) != set(DUAL_FACE_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return ["dual face requires all32 actual nonnegative integer fees"]
    if settled is None or settled["selection"]["canonical_exact"]:
        return [] if evidence is None and not any(work.values()) else ["no remaining root completion may export or charge a dual face"]
    if evidence is None or result["probe_stage"] != "root" or result["probe"] is None:
        return ["open settled roots require their paid remaining box and exact dual evidence"]
    selected, phase = settled["selection"], result["root_settle_work"]
    score, duration = _score(selected, queue), Fraction.from_float(selected["duration"])
    errors = []
    if (Fraction(*evidence["incumbent_score_fraction"]) != score or Fraction(*evidence["incumbent_duration_fraction"]) != duration
            or evidence["root_count"] != result["root_lp_budget"]
            or evidence["raw_region_count"] != phase["frontier_regions_after"]
            or evidence["live_region_count"] != selected["frontier_regions_remaining"]
            or len(evidence["regions"]) != evidence["live_region_count"]):
        errors.append("face coverage and incumbent must refer to the actual settled heap, rather than the raw root heap")
    grouped = {}
    durations = [Fraction.from_float(max(1., float(option["duration"]))) for option in model["options"]]
    for index, option in enumerate(model["options"]):
        grouped.setdefault(tuple(option["requirements"]), []).append(index)
    requirements = list(grouped)
    minima = [min(durations[index] for index in grouped[requirement]) for requirement in requirements]
    dimensions = len(queue)
    columns = positions = certified = cert_columns = tight_columns = fixed_lower = fixed_upper = price_copies = 0
    for region in evidence["regions"]:
        threshold = Fraction(*region["threshold_fraction"])
        if threshold not in minima or threshold > duration or Fraction(*region["bound_fraction"]) != score:
            errors.append("live child boxes must retain real duration strata and the certified incumbent score bound")
        reps = {group: next(index for index in grouped[required] if durations[index] <= threshold)
                for group, required in enumerate(requirements) if minima[group] <= threshold}
        eligible = sorted(reps, key=reps.get)
        req = [requirements[group] for group in eligible]
        exact = [pos for pos, group in enumerate(eligible) if minima[group] == threshold]
        if not all(_same(region[name], value) for name, value in (("eligible", eligible), ("requirements", req),
                ("representative_options", [reps[group] for group in eligible]), ("exact_duration_positions", exact))):
            errors.append("child-box groups, canonical original representatives or exact-duration positions differ from the catalogue")
        lower, upper = region["lower"], region["upper"]
        capacity = [min(lanes, *(waiting // amount for waiting, amount in zip(queue, required, strict=True) if amount)) for required in req]
        if (len(lower) != len(req) or len(upper) != len(req)
                or any(type(left) is not int or type(right) is not int or not 0 <= left <= right <= cap
                       for left, right, cap in zip(lower, upper, capacity, strict=True))
                or sum(lower) > lanes or any(sum(count * required[c] for count, required in zip(lower, req, strict=True)) > waiting
                                            for c, waiting in enumerate(queue))):
            errors.append("settled child bounds must remain inside the physical capacity box with feasible mandatory counts")
        columns += len(req)
        positions += len(exact)
        target = score * threshold
        if Fraction(*region["target_numerator_fraction"]) != target:
            errors.append("child face target differs from its exact incumbent score times its own duration")
        face_lower, face_upper = list(lower), list(upper)
        prices = region["certificate_prices"]
        if prices is None:
            if region["tight"] or region["own_numerator_fraction"] is not None or region["lower_coefficients"] or region["upper_coefficients"]:
                errors.append("uncertified child boxes cannot claim a tight dual or fixed support")
        else:
            certified += 1
            cert_columns += len(req)
            price_copies += len(prices)
            prices = [Fraction(*pair) for pair in prices]
            anchor_lower, anchor_upper = region["certificate_lower"], region["certificate_upper"]
            if (len(prices) != 1 + dimensions + len(req) or any(price < 0 for price in prices)
                    or not _same(anchor_lower, [0] * len(req)) or not _same(anchor_upper, capacity)):
                errors.append("child faces must retain the full nonnegative original-root dual and its physical anchor box")
                continue
            weights = [sum(q * a for q, a in zip(queue, required, strict=True)) for required in req]
            alpha = [weight - prices[0] - sum(prices[1 + c] * amount for c, amount in enumerate(required))
                     - prices[1 + dimensions + pos] for pos, (weight, required) in enumerate(zip(weights, req, strict=True))]
            upper_prices = prices[1 + dimensions:]
            if (any(coefficient > 0 for coefficient in alpha)
                    or not _same(region["lower_coefficients"], [(value.numerator, value.denominator) for value in alpha])
                    or not _same(region["upper_coefficients"], [(value.numerator, value.denominator) for value in upper_prices])):
                errors.append("child coefficients must equal independently recomputed exact feasible dual coefficients")
            anchor = sum(count * weight for count, weight in zip(anchor_lower, weights, strict=True))
            anchor += prices[0] * (lanes - sum(anchor_lower))
            anchor += sum(prices[1 + c] * (queue[c] - sum(count * required[c] for count, required in
                          zip(anchor_lower, req, strict=True))) for c in range(dimensions))
            anchor += sum(price * (right - left) for price, left, right in zip(upper_prices, anchor_lower, anchor_upper, strict=True))
            own = anchor + sum((new - old) * coefficient for new, old, coefficient in zip(lower, anchor_lower, alpha, strict=True))
            own += sum((new - old) * price for new, old, price in zip(upper, anchor_upper, upper_prices, strict=True))
            if (Fraction(*region["certificate_numerator_fraction"]) != anchor or Fraction(*region["own_numerator_fraction"]) != own
                    or own < target or region["tight"] != (own == target)):
                errors.append("child tightness must use its source certificate's own exact delta-adjusted numerator")
            if own == target:
                tight_columns += len(req)
                for pos, (coefficient, price) in enumerate(zip(alpha, upper_prices, strict=True)):
                    if coefficient < 0:
                        face_upper[pos] = lower[pos]
                        fixed_lower += 1
                    if price > 0:
                        face_lower[pos] = upper[pos]
                        fixed_upper += 1
        if not _same(region["face_lower"], face_lower) or not _same(region["face_upper"], face_upper):
            errors.append("only own-tight nonzero dual coefficients may fix a current child bound")
    boxes = len(evidence["regions"])
    key_checks, key_visits = work["canonical_key_comparisons"], work["canonical_key_variable_visits"]
    key_reads = work["canonical_key_count_reads"] - 2 * key_visits
    key_columns = key_reads // 3
    if (key_reads % 3 or not boxes <= key_checks <= evidence["raw_region_count"]
            or not columns <= key_columns <= len(grouped) * key_checks or not 0 <= key_visits <= key_columns):
        errors.append("canonical box scans must charge their actual variables and compared settled heap entries")
    completion = result["probe"]["completion"]
    visits, candidates = work["domain_box_visits"], work["candidate_key_comparisons"]
    ordered = sorted(evidence["regions"], key=lambda box: Fraction(*box["threshold_fraction"]))
    visited = ordered[:visits]
    if (not 0 <= candidates <= visits <= boxes
            or completion["thresholds_checked"] != len({Fraction(*box["threshold_fraction"]) for box in visited})):
        errors.append("completion must retain its actual box prefix, distinct durations and every found candidate")
    integer_ops, domain_reads, suffix_reads, noninteger, feasible_boxes = 2 * cert_columns, 0, 0, 0, 0
    for box in visited:
        if Fraction(*box["target_numerator_fraction"]).denominator != 1:
            noninteger += 1
            continue
        size = len(box["eligible"])
        caps = [right - left for left, right in zip(box["face_lower"], box["face_upper"], strict=True)]
        remaining = [waiting - sum(count * required[c] for count, required in
                     zip(box["face_lower"], box["requirements"], strict=True)) for c, waiting in enumerate(queue)]
        valid = min(caps, default=0) >= 0 and sum(box["face_lower"]) <= lanes and min(remaining, default=0) >= 0
        integer_ops += 4 * size * dimensions + 2 * size + 1 + dimensions + (2 * size if valid else 0)
        domain_reads += (3 + dimensions) * size
        suffix_reads += size * dimensions
        if valid:
            feasible_boxes += 1
            domain_reads += size + len(box["exact_duration_positions"])
            suffix_reads += 2 * dimensions * sum(cap > 0 for cap in caps)
    branches = completion["states_visited"] - sum(completion[name] for name in ("bound_prunes", "gcd_prunes", "duration_prunes")) - candidates
    if (branches < 0 or candidates > feasible_boxes or noninteger != completion["noninteger_thresholds"]
            or suffix_reads != completion["suffix_coordinate_reads"]):
        errors.append("boxed DFS states, skipped targets or suffix reads differ from the actual remaining domains")
    domain_reads += branches + (candidates * len(visited[-1]["eligible"]) if visited else 0)
    option_count = len(model["options"])
    integer_ops += candidates * (2 * option_count * dimensions + 2 * dimensions)
    extra_threshold = int(bool(candidates and visits < boxes and completion["complete"]
        and Fraction(*ordered[visits]["threshold_fraction"]) != Fraction(*visited[-1]["threshold_fraction"])))
    rational_base = 4 + 2 * boxes + (2 * dimensions + 7) * cert_columns + 2 * visits + extra_threshold + 3 * candidates
    # Multiple same-duration winners, including a discarded partial winner on
    # exhaustion, have separately paid supports that the returned action cannot
    # identify. Their aggregate conversion count must be physically possible.
    support_reads = work["exact_rational_operations"] - rational_base
    if not candidates <= support_reads <= candidates * min(option_count, lanes):
        errors.append("original candidate duration conversions must charge a feasible actual support for every found action")
    expected = dict(capture_calls=1, heap_entries_read=evidence["raw_region_count"], canonical_box_checks=evidence["raw_region_count"],
        canonical_key_integer_operations=key_columns + 2 * key_checks + 3 * key_visits,
        canonical_key_indices_written=lanes * (key_checks + int(key_checks > 0)), boxes_exported=boxes,
        group_ids_copied=columns, requirement_coordinates_copied=columns * dimensions, representative_options_copied=columns,
        box_bound_values_copied=2 * columns, exact_positions_copied=positions,
        fraction_pairs_copied=2 + 3 * boxes + price_copies + 2 * certified + 2 * cert_columns,
        certificate_price_reads=price_copies + (dimensions + 2) * cert_columns,
        certificate_bound_reads=4 * cert_columns, coefficient_requirement_reads=dimensions * cert_columns,
        exact_integer_operations=integer_ops, own_bound_comparisons=certified, coefficient_sign_comparisons=2 * tight_columns,
        fixed_lower_coordinates=fixed_lower, fixed_upper_coordinates=fixed_upper, face_bound_values_written=4 * columns + fixed_lower + fixed_upper,
        domain_bound_reads=domain_reads, domain_sort_calls=1, domain_sort_entries=boxes,
        candidate_option_visits=candidates * option_count, candidate_requirement_reads=candidates * option_count * dimensions)
    differing = [name for name, value in expected.items() if work[name] != value]
    if differing:
        errors.append("child-face fees differ from their actual evidence: " + ", ".join(differing))
    return errors


def _audit_settled_root_result(result, state, protocol, model, lanes, label, expected):
    if label != "candidate":
        metric, errors, quality = _audit_native_root_result(result, state, protocol, model, lanes,
            "candidate" if label == "native" else label, expected)
        if label == "native" and not _same(result, state["previous_native_root_routed"]):
            errors.append("native control differs from the actual frozen whole-selector parent")
        return metric, errors, quality
    errors = _settle_errors(result, state)
    root, settled = result["root_lp"], result["settled_lp"]
    roots = result["root_lp_budget"]
    triggered = root["selection"]["score_exact"] and not root["selection"]["canonical_exact"]
    if not triggered:
        metric, extra, quality = _audit_native_root_result(
            {name: value for name, value in result.items() if name not in SETTLE_FIELDS}, state, protocol, model, lanes, "candidate", expected)
        return metric, errors + extra, quality
    for name in ("initial_lp_budget", "lp_node_budget", "integer_state_budget"):
        if result[name] != protocol[name]:
            errors.append("settled candidate " + name + " differs from registration")
    errors.extend("raw root: " + message for message in audit_reserved_stage(root, roots, roots))
    if settled is None or result["root_settle_work"] is None:
        metric = _metric(result["selection"], state, model, lanes, result["final_lp"]["selection"]["lp_node_budget"], expected, result["final_lp"]["preprocessing"])
        return metric, errors, False
    errors.extend("settled: " + message for message in audit_root_stage(settled))
    previous = None
    for stage in (root, settled, result["initial_lp"], result["final_lp"]):
        errors.extend(audit_affine_stage(stage, len(state["queue"]), previous))
        errors.extend(audit_zero_stage(stage, previous))
        previous = stage
    errors.extend(audit_settled_dual_face(result, model, state["queue"], lanes))
    initial, final, probe = (result[name] for name in ("initial_lp", "final_lp", "probe"))
    if result["frontier_evidence"] is not None:
        errors.append("settled score-certified roots cannot introduce a late score-proof frontier")
    if settled["selection"]["canonical_exact"]:
        if (not _same(_without_cap(settled), _without_cap(state["previous_zero_traces"][str(protocol["initial_lp_budget"])]))
                or probe is not None or result["probe_stage"] != "none" or result["resumed"]
                or result["status"] != "LP_ROOT_SETTLE_COMPLETE" or not _same(initial, settled)
                or not _same(final, settled) or not _same(result["selection"], settled["selection"])):
            errors.append("closed settlement must return the genuine zero native prefix at M without an integer probe")
    elif probe is not None and result["probe_stage"] == "root":
        try:
            _, extra, _ = _audit_strict(probe, state, {**protocol, "lp_budgets": {"candidate": roots}}, model, lanes, "candidate", expected)
            errors.extend("settled root probe: " + message for message in extra)
        except ValueError as error:
            errors.append("settled root probe: " + str(error))
        if any(not _same(probe[field], settled[source]) for field, source in (
                ("baseline_selection", "selection"), ("preprocessing", "preprocessing"), ("capacity_propagation", "capacity_propagation"))):
            errors.append("the unique root probe must preserve its actual settled baseline and paid native work")
        if probe["selection"]["canonical_exact"]:
            if (result["resumed"] or result["status"] != "INTEGER_PROBE_COMPLETE" or not _same(initial, settled)
                    or not _same(final, settled) or not _same(result["selection"], probe["selection"])):
                errors.append("successful settled completion must return its original action at the actual M checkpoint")
        else:
            if (probe["completion"]["status"] != "STATE_BUDGET_EXHAUSTED"
                    or probe["score_proof"]["states_visited"] + probe["completion"]["states_visited"] != protocol["integer_state_budget"]
                    or not _same(probe["selection"], settled["selection"])):
                errors.append("exhausted root completion must retain its entire sole budget and actual settled baseline")
            should_resume = not initial["selection"]["canonical_exact"]
            cap = protocol["lp_node_budget"] if should_resume else protocol["initial_lp_budget"]
            status = ("LP_RESUME_COMPLETE" if final["selection"]["canonical_exact"] else "LP_RESUME_INCOMPLETE") if should_resume else "LP_INITIAL_COMPLETE"
            if (not _same(initial, state["previous_zero_traces"][str(protocol["initial_lp_budget"])] )
                    or not _same(final, state["previous_zero_traces"][str(cap)]) or result["resumed"] != should_resume
                    or result["status"] != status or not _same(result["selection"], final["selection"])):
                errors.append("root exhaustion permits only the genuine same-search144/256 continuation, with no later probe")
    else:
        errors.append("open settlement requires its retained sole root probe")
    budget = final["selection"]["lp_node_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, final["preprocessing"])
    if result["selection"]["lp_node_budget"] != budget or metric["failures"]:
        errors.append("returned settled action has invalid feasibility, certificate or actual cumulative LP work")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(results, label):
    work = _native_work(results, "candidate" if label == "native" else label)
    if label == "candidate":
        phases = [result["root_settle_work"] for result in results if result["root_settle_work"] is not None]
        work["root_settle_work"] = {name: sum(phase[name] for phase in phases) for name in ROOT_SETTLE_FEE_COUNTERS}
        work["root_settle_diagnostics"] = {name: sum(phase[name] for phase in phases)
            for name in ROOT_SETTLE_COUNTERS if name not in ROOT_SETTLE_FEE_COUNTERS}
    return work


def summarize(protocol, rows):
    # Validate the real parent and all real borrowed ladders/references once;
    # every current action, phase and changed completion is audited separately.
    parent_rows = [{**row, "states": [{**state, "routed": state["previous_native_root_routed"]}
                                     for state in row["states"]]} for row in rows]
    result = _native_summary(protocol, parent_rows)
    result["source_native_behavior_preserved"] = result.pop("behavior_preserved")
    models = {model["node"]: model for model in protocol["models"]}
    result["nonclosures"] = []
    entries = [(row, state) for row in rows for state in row["states"]]
    for row in rows:
        actual_calls = sum(state["routed"]["root_settle_work"] is not None for state in row["states"])
        if row["root_settle_calls"] != actual_calls:
            result["failures"].append({"node": row["node"], "lanes": row["lanes"], "errors": ["actual row root-settle call coverage differs"]})
        model = models[row["node"]]
        for state in row["states"]:
            pointer = {"node": row["node"], "lanes": row["lanes"], **{name: state[name] for name in ("state_id", "queue", "timing_group")}}
            expected, _ = _reference(state, protocol, len(model["options"]))
            _, errors, quality = _audit_settled_root_result(state["routed"], state, protocol, model, row["lanes"], "candidate", expected)
            if errors:
                result["failures"].append({**pointer, "errors": errors})
            if not quality:
                result["nonclosures"].append({**pointer, "status": state["routed"]["status"]})
    states = [state for _, state in entries]
    actual_calls = sum(state["routed"]["root_settle_work"] is not None for state in states)
    if actual_calls != protocol["expected_root_settle_calls"]:
        result["failures"].append({"errors": ["every registered raw score-certified canonical-open root must settle once"]})
    executed = _work([state["routed"] for state in states], "candidate")
    result["executed"]["whole_selector"] = executed
    result["executed_routing_preparation_work"] = executed["routing_preparation"]
    result["executed_dual_face_work"] = executed["dual_face_work"]
    result["executed_root_settle_work"] = executed["root_settle_work"]
    result["root_settle_diagnostics"] = executed["root_settle_diagnostics"]
    result["executed_root_settle_calls"] = actual_calls
    result["borrowed"]["whole_selector"]["native"] = _work([state["previous_native_root_routed"] for state in states], "native")
    result["borrowed_native_root_snapshots"] = sum(len(state["previous_native_root_traces"]) for state in states)
    for name, group in result["groups"].items():
        items = [state for state in states if state["timing_group"] == name]
        group["executed"]["whole_selector"] = _work([state["routed"] for state in items], "candidate")
        group["borrowed"]["whole_selector"]["native"] = _work([state["previous_native_root_routed"] for state in items], "native")
    result["root_probe_states"] = [{"node": row["node"], "lanes": row["lanes"], **{name: state[name] for name in ("state_id", "queue")},
        "completion_status": state["routed"]["probe"]["completion"]["status"], "work": _work([state["routed"]], "candidate")}
        for row, state in entries if state["routed"]["probe_stage"] == "root"]
    result["root_settle_states"] = [{"node": row["node"], "lanes": row["lanes"], **{name: state[name] for name in ("state_id", "queue")},
        "status": state["routed"]["status"], "work": _work([state["routed"]], "candidate")}
        for row, state in entries if state["routed"]["root_settle_work"] is not None]
    execution = not result["failures"]
    advance = bool(execution and result["references_complete"] and not result["nonclosures"])
    result.update(execution_checks_passed=execution, work_gate_passed=execution, quality_gate_passed=advance,
        development_gate_passed=advance, decision="INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not result["references_complete"] else
            "SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE" if advance else "SETTLED_ROOT_DUAL_FACE_DEVELOPMENT_HOLD")
    return result
