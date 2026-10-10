"""Audit the paid root boxes, their own exact dual face and all executed work."""
from __future__ import annotations

from fractions import Fraction

from algorithm.experiments.early_root_canonical_development_summary import (
    ROUTING_COUNTERS, _audit_early_result, _work as _early_work, summarize as _early_summary,
)
from algorithm.experiments.strict_score_proof_timing_summary import _same

DUAL_FACE_COUNTERS = (
    "capture_calls", "heap_entries_read", "canonical_box_checks", "canonical_key_variable_visits",
    "canonical_key_count_reads", "canonical_key_integer_operations", "canonical_key_indices_written",
    "canonical_key_comparisons", "boxes_exported", "group_ids_copied", "requirement_coordinates_copied",
    "representative_options_copied", "box_bound_values_copied", "exact_positions_copied", "fraction_pairs_copied",
    "certificate_price_reads", "certificate_bound_reads", "coefficient_requirement_reads",
    "exact_integer_operations", "exact_rational_operations", "own_bound_comparisons",
    "coefficient_sign_comparisons", "fixed_lower_coordinates", "fixed_upper_coordinates",
    "face_bound_values_written", "domain_box_visits", "domain_bound_reads", "domain_sort_calls",
    "domain_sort_entries", "candidate_option_visits", "candidate_requirement_reads", "candidate_key_comparisons",
)
FACE_FIELDS = ("dual_face_evidence", "dual_face_work")


def audit_dual_face(result, model, queue, lanes):
    """Recompute exact dual algebra from the catalogue and returned evidence."""
    work = result.get("dual_face_work", {})
    if set(work) != set(DUAL_FACE_COUNTERS) or any(type(value) is not int or value < 0 for value in work.values()):
        return ["dual face requires all32 actual nonnegative integer fees"]
    root, evidence = result["root_lp"], result["dual_face_evidence"]
    selected = root["selection"]
    triggered = selected["score_exact"] and not selected["canonical_exact"]
    if not triggered:
        return [] if evidence is None and not any(work.values()) else ["untriggered face must export no evidence and charge zero face work"]
    if evidence is None or result["probe_stage"] != "root":
        return ["root known-score completion requires its paid box and exact dual evidence"]
    score = Fraction(sum(q * s for q, s in zip(queue, selected["service"], strict=True))) / Fraction.from_float(selected["duration"])
    errors = []
    if (Fraction(*evidence["incumbent_score_fraction"]) != score
            or Fraction(*evidence["incumbent_duration_fraction"]) != Fraction.from_float(selected["duration"])
            or evidence["root_count"] != result["root_lp_budget"]
            or evidence["live_region_count"] != selected["frontier_regions_remaining"]
            or len(evidence["regions"]) != evidence["live_region_count"]
            or 2 * evidence["raw_region_count"] != result["routing_preparation"]["root_snapshot_heap_entries_read"]):
        errors.append("face incumbent, roots or complete paid root-box coverage differs from the raw root snapshot")
    grouped = {}
    durations = [Fraction.from_float(max(1., float(option["duration"]))) for option in model["options"]]
    for index, option in enumerate(model["options"]):
        grouped.setdefault(tuple(option["requirements"]), []).append(index)
    requirements = list(grouped)
    minima = [min(durations[index] for index in grouped[requirement]) for requirement in requirements]
    dimensions, columns, positions, seen = len(queue), 0, 0, set()
    certified = cert_columns = tight_columns = fixed_lower = fixed_upper = 0
    price_copies = 0
    for region in evidence["regions"]:
        threshold, bound = Fraction(*region["threshold_fraction"]), Fraction(*region["bound_fraction"])
        if (threshold not in minima or threshold in seen or bound != score
                or threshold > Fraction(*evidence["incumbent_duration_fraction"])):
            errors.append("live roots must use distinct real duration strata at the certified incumbent score")
        seen.add(threshold)
        reps = {group: next(index for index in grouped[requirement] if durations[index] <= threshold)
                for group, requirement in enumerate(requirements) if minima[group] <= threshold}
        eligible = sorted(reps, key=reps.get)
        req = [requirements[group] for group in eligible]
        exact = [pos for pos, group in enumerate(eligible) if minima[group] == threshold]
        if not all(_same(region[field], expected) for field, expected in (
                ("eligible", eligible), ("requirements", req), ("representative_options", [reps[group] for group in eligible]),
                ("exact_duration_positions", exact))):
            errors.append("face groups, original canonical representatives or exact-duration positions differ from the catalogue")
        lower, upper = region["lower"], region["upper"]
        capacity = [min(lanes, *(waiting // amount for waiting, amount in zip(queue, required, strict=True) if amount)) for required in req]
        if not _same(lower, [0] * len(req)) or not _same(upper, capacity):
            errors.append("exported unbranched roots must retain their full capacity-feasible count boxes")
        columns += len(req)
        positions += len(exact)
        target = score * threshold
        if Fraction(*region["target_numerator_fraction"]) != target:
            errors.append("face target numerator differs from its own root score times its own duration")
        face_lower, face_upper = list(lower), list(upper)
        prices = region["certificate_prices"]
        if prices is None:
            if region["tight"] or region["own_numerator_fraction"] is not None or region["lower_coefficients"] or region["upper_coefficients"]:
                errors.append("an uncertified root box cannot claim a dual face or fixed count support")
        else:
            certified += 1
            cert_columns += len(req)
            price_copies += len(prices)
            prices = [Fraction(*pair) for pair in prices]
            anchor_lower, anchor_upper = region["certificate_lower"], region["certificate_upper"]
            if len(prices) != 1 + dimensions + len(req) or any(price < 0 for price in prices):
                errors.append("face prices must be a full exactly feasible nonnegative max-LP dual")
                continue
            weights = [sum(q * a for q, a in zip(queue, required, strict=True)) for required in req]
            alpha = [weight - prices[0] - sum(prices[1 + coordinate] * amount for coordinate, amount in enumerate(required))
                     - prices[1 + dimensions + pos] for pos, (weight, required) in enumerate(zip(weights, req, strict=True))]
            upper_prices = prices[1 + dimensions:]
            if any(coefficient > 0 for coefficient in alpha) or not _same(
                    region["lower_coefficients"], [(value.numerator, value.denominator) for value in alpha]) or not _same(
                    region["upper_coefficients"], [(value.numerator, value.denominator) for value in upper_prices]):
                errors.append("face lower coefficients or upper prices differ from the independent exact dual feasibility calculation")
            own_at_anchor = sum(count * weight for count, weight in zip(anchor_lower, weights, strict=True))
            own_at_anchor += prices[0] * (lanes - sum(anchor_lower))
            own_at_anchor += sum(prices[1 + coordinate] * (queue[coordinate] - sum(
                count * required[coordinate] for count, required in zip(anchor_lower, req, strict=True))) for coordinate in range(dimensions))
            own_at_anchor += sum(price * (right - left) for price, left, right in zip(upper_prices, anchor_lower, anchor_upper, strict=True))
            own = own_at_anchor + sum((new - old) * coefficient for new, old, coefficient in zip(lower, anchor_lower, alpha, strict=True))
            own += sum((new - old) * price for new, old, price in zip(upper, anchor_upper, upper_prices, strict=True))
            if (Fraction(*region["certificate_numerator_fraction"]) != own_at_anchor
                    or Fraction(*region["own_numerator_fraction"]) != own or own < target or region["tight"] != (own == target)):
                errors.append("face must use its certificate's own exact numerator and equality, never the inherited minimum bound")
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
            errors.append("only the certificate's own tight nonzero dual coefficients may fix its original count bounds")
    boxes = len(evidence["regions"])
    key_checks, key_visits = work["canonical_key_comparisons"], work["canonical_key_variable_visits"]
    key_reads = work["canonical_key_count_reads"] - 2 * key_visits
    key_columns = key_reads // 3
    if (key_reads % 3 or not boxes <= key_checks <= evidence["raw_region_count"]
            or not columns <= key_columns <= len(grouped) * key_checks or not 0 <= key_visits <= key_columns):
        errors.append("optimistic canonical-key scans must charge every actual variable and compared raw root")
    probe = result["probe"]
    completion = probe["completion"]
    visits, candidates = work["domain_box_visits"], work["candidate_key_comparisons"]
    if (visits != completion["thresholds_checked"] or not 0 <= visits <= boxes or candidates not in (0, 1)
            or candidates > visits):
        errors.append("root domain preparation must retain its actual distinct-duration prefix and candidate count")
    visited = sorted(evidence["regions"], key=lambda box: Fraction(*box["threshold_fraction"]))[:visits]
    integer_ops = 2 * cert_columns
    domain_reads = suffix_reads = noninteger = 0
    for box in visited:
        if Fraction(*box["target_numerator_fraction"]).denominator != 1:
            noninteger += 1
            continue
        size = len(box["eligible"])
        caps = [right - left for left, right in zip(box["face_lower"], box["face_upper"], strict=True)]
        remaining = [waiting - sum(count * required[coordinate] for count, required in
            zip(box["face_lower"], box["requirements"], strict=True)) for coordinate, waiting in enumerate(queue)]
        valid = min(caps, default=0) >= 0 and sum(box["face_lower"]) <= lanes and min(remaining, default=0) >= 0
        integer_ops += 4 * size * dimensions + 2 * size + 1 + dimensions + (2 * size if valid else 0)
        domain_reads += (3 + dimensions) * size
        suffix_reads += size * dimensions
        if valid:
            domain_reads += size + len(box["exact_duration_positions"])
            suffix_reads += 2 * dimensions * sum(cap > 0 for cap in caps)
    branches = completion["states_visited"] - sum(completion[name] for name in ("bound_prunes", "gcd_prunes", "duration_prunes")) - candidates
    if branches < 0 or noninteger != completion["noninteger_thresholds"] or suffix_reads != completion["suffix_coordinate_reads"]:
        errors.append("boxed DFS states, threshold skips or suffix reads differ from their paid root domains")
    domain_reads += branches + (len(visited[-1]["eligible"]) if candidates and visited else 0)
    option_count = len(model["options"])
    integer_ops += candidates * (2 * option_count * dimensions + 2 * dimensions)
    rational_base = (4 + 2 * boxes + (2 * dimensions + 7) * cert_columns + 2 * visits
                     + int(bool(candidates and visits < boxes)) + 3 * candidates)
    active = candidates * sum(count > 0 for count in probe["selection"]["option_counts"])
    if work["exact_rational_operations"] != rational_base + active:
        errors.append("exact rational arithmetic must include every certificate, visited duration and used original candidate option")
    expected_fees = dict(capture_calls=1, heap_entries_read=evidence["raw_region_count"], canonical_box_checks=evidence["raw_region_count"],
        canonical_key_integer_operations=key_columns + 2 * key_checks + 3 * key_visits,
        canonical_key_indices_written=lanes * (key_checks + int(key_checks > 0)),
        boxes_exported=boxes, group_ids_copied=columns,
        requirement_coordinates_copied=columns * dimensions, representative_options_copied=columns,
        box_bound_values_copied=2 * columns, exact_positions_copied=positions,
        fraction_pairs_copied=2 + 3 * boxes + price_copies + 2 * certified + 2 * cert_columns,
        certificate_price_reads=price_copies + (dimensions + 2) * cert_columns,
        certificate_bound_reads=4 * cert_columns, coefficient_requirement_reads=dimensions * cert_columns,
        exact_integer_operations=integer_ops, coefficient_sign_comparisons=2 * tight_columns,
        fixed_lower_coordinates=fixed_lower, fixed_upper_coordinates=fixed_upper,
        face_bound_values_written=4 * columns + fixed_lower + fixed_upper, own_bound_comparisons=certified,
        domain_bound_reads=domain_reads, domain_sort_calls=1, domain_sort_entries=boxes,
        candidate_option_visits=candidates * option_count, candidate_requirement_reads=candidates * option_count * dimensions)
    different = [name for name, value in expected_fees.items() if work[name] != value]
    if different:
        errors.append("face operation fees differ from their actual evidence: " + ", ".join(different))
    return errors


def _face_errors(result, state, model, lanes):
    errors = audit_dual_face(result, model, state["queue"], lanes)
    root = result["root_lp"]["selection"]
    if not (root["score_exact"] and not root["canonical_exact"]) and not _same(
            {name: value for name, value in result.items() if name not in FACE_FIELDS}, state["previous_early_routed"]):
        errors.append("untriggered root face must preserve every actual early route field and fee")
    return errors


def _audit_dual_face_result(result, state, protocol, model, lanes, label, expected):
    metric, errors, quality = _audit_early_result(result, state, protocol, model, lanes,
        "candidate" if label == "early" else label, expected)
    if label == "early":
        if not _same(result, state["previous_early_routed"]):
            errors.append("early control differs from its actual frozen whole-selector result")
    elif label == "candidate":
        errors.extend(_face_errors(result, state, model, lanes))
    return metric, errors, quality


def _work(results, label):
    work = _early_work(results, "candidate" if label == "early" else label)
    if label == "candidate":
        work["dual_face_work"] = {name: sum(result["dual_face_work"][name] for result in results)
                                  for name in DUAL_FACE_COUNTERS}
    return work


def summarize(protocol, rows):
    # The public early record remains unchanged. Its frozen audit checks every
    # raw LP snapshot and the unique integer probe without normalizing costs.
    result = _early_summary(protocol, rows)
    models = {model["node"]: model for model in protocol["models"]}
    states = [state for row in rows for state in row["states"]]
    for row in rows:
        model = models[row["node"]]
        for state in row["states"]:
            errors = _face_errors(state["routed"], state, model, row["lanes"])
            if errors:
                result["failures"].append({"node": row["node"], "lanes": row["lanes"],
                    **{key: state[key] for key in ("state_id", "queue", "timing_group")}, "errors": errors})
    executed = _work([state["routed"] for state in states], "candidate")
    result["executed"]["whole_selector"] = executed
    result["executed_dual_face_work"] = executed["dual_face_work"]
    result["borrowed"]["whole_selector"]["early"] = _work([state["previous_early_routed"] for state in states], "early")
    for name, group in result["groups"].items():
        items = [state for state in states if state["timing_group"] == name]
        group["executed"]["whole_selector"] = _work([state["routed"] for state in items], "candidate")
        group["borrowed"]["whole_selector"]["early"] = _work([state["previous_early_routed"] for state in items], "early")
    root_states = [state for state in states if state["routed"]["probe_stage"] == "root"]
    for entry, state in zip(result["root_probe_states"], root_states, strict=True):
        entry["work"] = _work([state["routed"]], "candidate")
    execution = not result["failures"]
    advance = execution and result["references_complete"] and not result["nonclosures"]
    result.update(execution_checks_passed=execution, work_gate_passed=execution,
        quality_gate_passed=bool(advance), development_gate_passed=bool(advance),
        decision="INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not result["references_complete"] else
            "DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_ADVANCE" if advance else "DUAL_FACE_ROOT_CANONICAL_DEVELOPMENT_HOLD")
    return result
