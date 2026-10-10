"""Borrowed-roster axis proof development with same-arm work comparisons."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from types import SimpleNamespace

from algorithm.axis_capacity_frontier_score_proof_frame_candidates import CAPACITY_COUNTERS as AXIS_COUNTERS
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection, selection_score
from algorithm.experiments.branch_reserved_development_summary import _identity, _key, audit_reserved_stage
from algorithm.experiments.dual_reuse_development_summary import _dual_work, audit_dual_stage
from algorithm.experiments.frontier_score_proof_confirmation_summary import audit_frontier_confirmation_router
from algorithm.experiments.frontier_score_proof_development_summary import _work as _frontier_work
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.resumable_probe_development_summary import _candidate_work
from algorithm.experiments.root_pruning_development_summary import _native_work
from algorithm.experiments.score_face_completion_confirmation_summary import _reference, ORACLE_COUNTERS
from algorithm.experiments.strict_score_proof_timing_summary import _same


def _groups(evidence):
    groups = {}
    for index, region in enumerate(evidence["regions"]):
        key = (tuple(region["threshold_fraction"]), tuple(region["eligible"]), tuple(map(tuple, region["requirements"])))
        groups.setdefault(key, []).append(index)
    return [(key, indices) for key, indices in groups.items() if len(indices) >= 2]


def _profile_upper(profiles, queue, lanes):
    merged = [0] + [-1] * lanes
    for waiting, profile in zip(queue, profiles):
        merged = [max((merged[slots - used] + waiting * profile[used]
            for used in range(slots + 1) if merged[slots - used] >= 0 and profile[used] >= 0), default=-1)
            for slots in range(lanes + 1)]
    return max(merged)


def audit_axis_capacity_proof(probe, evidence, queue, lanes):
    proof, errors = probe["score_proof"], []
    if "capacity" not in proof or "dfs_states_visited" not in proof:
        return ["frontier proof must retain its actual DP and fallback DFS work"]
    work, dimensions = proof["capacity"], len(queue)
    if any(type(work[key]) is not int or work[key] < 0 for key in AXIS_COUNTERS) or type(proof["dfs_states_visited"]) is not int or proof["dfs_states_visited"] < 0:
        errors.append("axis and fallback work counters must be nonnegative integers")
    if (work["states_visited"] != work["axis_states"] + work["merge_states"]
            or proof["states_visited"] != work["states_visited"] + proof["dfs_states_visited"]):
        errors.append("shared proof states must include every DP axis/merge and fallback DFS state")
    groups = _groups(evidence)
    pure = [(key, indices) for key, indices in groups if all(sum(value > 0 for value in requirement) <= 1 for requirement in key[2])]
    bounds = work["bounds"]
    actual = [((tuple(bound["threshold_fraction"]), tuple(bound["eligible"]), tuple(map(tuple, bound["requirements"]))), list(bound["box_indices"])) for bound in bounds]
    if actual != pure[:len(bounds)] or work["threshold_tables"] != len(bounds):
        errors.append("completed capacity tables must cover the original repeated pure-axis groups in order")
    reads = 0
    completed = 0
    for key, indices in groups:
        for requirement in key[2]:
            reads += dimensions
            if sum(value > 0 for value in requirement) > 1:
                break
        else:
            if completed == len(bounds) and work["status"] == "STATE_BUDGET_EXHAUSTED":
                break
            completed += 1
    if work["requirement_coordinate_reads"] != reads:
        errors.append("axis eligibility scans must retain their actual requirement-coordinate reads")
    pruned = checks = hits = positive = 0
    for bound in bounds:
        target = Fraction(*evidence["incumbent_score_fraction"]) * Fraction(*bound["threshold_fraction"])
        target = target.numerator // target.denominator + 1
        profiles = bound["axis_service_profiles"]
        profile_valid = (len(profiles) == dimensions and all(len(profile) == lanes + 1 and profile[0] == 0
            and all(type(value) is int and -1 <= value <= waiting for value in profile) for profile, waiting in zip(profiles, queue)))
        if not profile_valid or type(bound["upper_numerator"]) is not int or bound["upper_numerator"] < 0:
            errors.append("capacity tables must contain exact-slot service profiles and an integer upper numerator")
        elif bound["upper_numerator"] != _profile_upper(profiles, queue, lanes):
            errors.append("capacity upper numerator differs from the recorded shared-slot profile merge")
        if bound["target_numerator"] != target:
            errors.append("capacity target must be the strict integer target of its own incumbent and stratum")
        size = len(bound["box_indices"])
        checks += size
        hits += size - 1
        pruned += size if bound["upper_numerator"] < target else 0
        positive += sum(any(value > 0 for value in requirement) for requirement in bound["requirements"])
    if (work["bound_checks"] != checks or work["cache_hits"] != hits or work["boxes_pruned"] != pruned
            or proof["bound_prunes"] < pruned or proof["thresholds_checked"] < pruned):
        errors.append("cached table checks and pruned boxes differ from the actual bound certificates")
    status = work["status"]
    if status == "NOT_ATTEMPTED":
        if work["attempted"] or bounds or any(work[key] for key in AXIS_COUNTERS if key != "requirement_coordinate_reads"):
            errors.append("a skipped DP must preserve the original DFS and charge only any mixed-axis scans")
    elif status == "COMPLETE":
        if not work["attempted"] or len(bounds) != len(pure):
            errors.append("completed DP must retain every attempted pure-axis table")
    elif status == "STATE_BUDGET_EXHAUSTED":
        if (not work["attempted"] or proof["status"] != status or proof["dfs_states_visited"] != 0
                or work["states_visited"] != probe["integer_state_budget"] or len(bounds) >= len(pure)):
            errors.append("DP exhaustion must consume the shared cap without fallback or a fabricated completed table")
    else:
        errors.append("unsupported axis-capacity outcome")
    tables = len(bounds)
    fees = {"axis_states": dimensions * tables + positive * (lanes + 1), "merge_states": tables * (1 + dimensions * (lanes + 1)),
        "bitset_shifts": dimensions * tables + positive * lanes, "bitset_ors": positive * (lanes + 1),
        "bitset_masks": positive * (lanes + 1), "service_reads": dimensions * tables * (lanes + 1),
        "merge_candidates": dimensions * tables * (lanes + 1) * (lanes + 2) // 2}
    if any(work[key] < value if status == "STATE_BUDGET_EXHAUSTED" else work[key] != value for key, value in fees.items()):
        errors.append("axis/merge states and explicit bitset/read/candidate fees omit computed table cells")
    return errors


def _work(states):
    work = _frontier_work(states)
    for label, field in (("old", "routed_old"), ("new", "routed_new")):
        result_field = "old_routed" if label == "old" else "routed"
        proofs = [state[result_field]["probe"]["score_proof"] for state in states if state[result_field]["probe"] is not None]
        capacities = [proof["capacity"] for proof in proofs if "capacity" in proof]
        work[field]["axis_capacity"] = {"attempted_calls": sum(capacity["attempted"] for capacity in capacities),
            "statuses": dict(Counter(capacity["status"] for capacity in capacities)),
            **{key: sum(capacity[key] for capacity in capacities) for key in AXIS_COUNTERS}}
        work[field]["fallback_dfs_states"] = sum(proof.get("dfs_states_visited", proof["states_visited"]) for proof in proofs)
    return work


def _prior_work(states):
    return {label: _candidate_work([state["prior_routed"][label] for state in states]) for label in ("old", "new")}


def summarize(protocol, rows):
    failures, nonclosures, regressions, comparisons, counterexamples, entries, cells = [], [], [], [], [], [], []
    qualities, complete = {}, True
    all_states = [state for row in rows for state in row["states"]]
    count = len(all_states)
    counts = {"expected_state_count": count, "expected_nonzero_state_count": sum(any(state["queue"]) for state in all_states),
        "expected_failure_development_count": sum(state["cohort"] == "failure_development" for state in all_states),
        "expected_axis_failure_development_count": sum(state["cohort"] == "axis_failure_development" for state in all_states),
        "expected_selector_calls": count * 2, "expected_interface_calls": count * 2, "expected_native_constructions": count * 2,
        "expected_native_advance_calls": 0,
        "expected_borrowed_native_snapshots": sum(len(state["traces"][label]) for state in all_states for label in ("old", "new"))}
    if any(protocol[key] != value for key, value in counts.items()):
        failures.append({"errors": ["complete borrowed roster or executed interface counts differ from registration"]})
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        failures.append({"errors": ["all registered cells must occur once in order"]})
    fixed = {name: {_key(item["node"], item["lanes"], item) for item in protocol[field]} for name, field in (
        ("historical", "historical_regression_keys"), ("failure_development", "failure_development_keys"), ("axis_failure_development", "axis_failure_development_keys"))}
    registered = {_key(cell["node"], cell["lanes"], state) for cell in protocol["cells"] for state in cell["states"]}
    if any(not keys <= registered for keys in fixed.values()):
        failures.append({"errors": ["all fixed historical and failure controls must be retained"]})
    models = {model["node"]: model for model in protocol["models"]}
    for cell, row in zip(protocol["cells"], rows):
        model, point = models[cell["node"]], {"node": row["node"], "lanes": row["lanes"]}
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        identity = lambda state: (_identity(state), tuple(state["caps"]), tuple(state[key] for key in (
            "borrowed_from", "borrowed_index", "source_state_id", "source_record_source")))
        errors = []
        if list(map(identity, row["states"])) != list(map(identity, cell["states"])):
            errors.append("state metadata or borrowed source pointers differ from registration")
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options)):
            errors.append("cell registration or catalogue metadata differs")
        if row["new_reference_calls"] or row["new_reference_oracle_calls"] or row["new_old_routed_calls"] != len(row["states"]):
            errors.append("references/native evidence must be borrowed and only both new cold arms executed")
        if any(row[key]["cpu_affinity"] != [protocol["cpu"]] or not row[key]["gc_enabled"] for key in ("runtime_start", "runtime_end")):
            errors.append("runtime affinity or GC differs from registration")
        if errors:
            failures.append({**point, "errors": errors})
        minima = {}
        for option in options:
            requirement = tuple(option.requirements)
            minima[requirement] = min(minima.get(requirement, float("inf")), max(1., option.duration))
        roots = len(set(minima.values()))
        cell_entries = []
        for state in row["states"]:
            pointer = {**point, **{key: state[key] for key in ("state_id", "queue", "source", "cohort", "historical_regression")}, "seed": state.get("seed")}
            errors, expected = [], None
            key = _key(row["node"], row["lanes"], state)
            if (state["historical_regression"] != (key in fixed["historical"])
                    or any((state["cohort"] == name) != (key in fixed[name]) for name in ("failure_development", "axis_failure_development"))
                    or state["borrowed_from"] not in protocol["borrowed_sources"]):
                errors.append("historical/failure cohort or borrowed source differs from its fixed control keys")
            if tuple(state["caps"]) != (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"]):
                errors.append("borrowed native ladder differs from the catalogue and registered budgets")
            reference = selection_from_record(state["reference"])
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            metric = state["reference_metrics"]
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"] or metric["failures"]
                    or metric["reference_comparable_count"] != int(state["reference_complete"])
                    or any(metric["search_work"][key] != state["reference"][key] for key in COUNTERS)):
                errors.append("borrowed reference has invalid budget, certificate or paid-work metrics")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options, tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            if not all(set(state["traces"][label]) == {str(cap) for cap in state["caps"]} for label in ("old", "new")):
                failures.append({**pointer, "errors": errors + ["all eight borrowed native snapshots must remain available"]})
                continue
            state_quality = {}
            for label, field in (("old", "old_routed"), ("new", "routed")):
                result, prior = state[field], state["prior_routed"][label]
                _, extra, quality = audit_frontier_confirmation_router(result, state, protocol, model, row["lanes"], label, expected)
                errors.extend(label + ": " + message for message in extra)
                if (not _same(result["initial_lp"], prior["initial_lp"]) or not _same(result["final_lp"], prior["final_lp"])
                        or result["resumed"] != prior["resumed"] or not _same(result["frontier_evidence"], prior["frontier_evidence"])):
                    errors.append(label + " paid LP, resumption or frontier export changed from its same-arm predecessor")
                if prior["probe_stage"] != "frontier_late":
                    if not _same(result, prior):
                        errors.append(label + " ordinary route must reproduce its complete previous output")
                elif result["probe"] is not None and result["frontier_evidence"] is not None:
                    errors.extend(label + ": " + message for message in audit_axis_capacity_proof(result["probe"], result["frontier_evidence"], state["queue"], row["lanes"]))
                for name, cap in (("initial_lp", 144), ("final_lp", 256 if result["resumed"] else 144)):
                    errors.extend(label + ": " + message for message in audit_reserved_stage(result[name], cap, roots))
                if label == "new":
                    errors.extend(audit_dual_stage(result["initial_lp"], len(state["queue"])))
                    errors.extend(audit_dual_stage(result["final_lp"], len(state["queue"]), result["initial_lp"]))
                qualities[row["node"], row["lanes"], state["state_id"], label] = quality
                state_quality[label] = quality
                if not quality:
                    nonclosures.append({**pointer, "label": label, "status": result["status"], "reference_complete": expected is not None})
                a, b = (_candidate_work([record]) for record in (prior, result))
                if (a["search_work"]["lp_calls"] != b["search_work"]["lp_calls"]
                        or a["search_work"]["lp_matrix_coefficients"] != b["search_work"]["lp_matrix_coefficients"]
                        or b["shared_integer_states"] > a["shared_integer_states"]):
                    regressions.append({**pointer, "label": label, "prior_integer_states": a["shared_integer_states"], "actual_integer_states": b["shared_integer_states"],
                        "prior_lp_calls": a["search_work"]["lp_calls"], "actual_lp_calls": b["search_work"]["lp_calls"],
                        "prior_matrix_coefficients": a["search_work"]["lp_matrix_coefficients"], "actual_matrix_coefficients": b["search_work"]["lp_matrix_coefficients"]})
            for cap in state["caps"]:
                scores = [selection_score(selection_from_record(state["traces"][label][str(cap)]["selection"]), tuple(state["queue"])) for label in ("old", "new")]
                comparisons.append({**pointer, "lp_budget": cap, "old_score": str(scores[0]), "new_score": str(scores[1]), "score_regression": scores[1] < scores[0]})
            if state["cohort"] in ("failure_development", "axis_failure_development"):
                counterexamples.append({**pointer, "quality": state_quality,
                    "prior_status": {label: state["prior_routed"][label]["status"] for label in ("old", "new")},
                    "status": {label: state[field]["status"] for label, field in (("old", "old_routed"), ("new", "routed"))}, "work": _work([state])})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            cell_entries.append(state)
        cells.append({**point, "work": _work(cell_entries), "prior_work": _prior_work(cell_entries)})
    groups = {}
    cohorts = defaultdict(list)
    for state in entries:
        cohorts[state["cohort"]].append(state)
    for cohort, selected in cohorts.items():
        groups[cohort] = {"state_count": len(selected), "work": _work(selected), "prior_work": _prior_work(selected),
            "quality_gate_passed": not any(item["cohort"] == cohort for item in nonclosures),
            "predecessor_work_gate_passed": not any(item["cohort"] == cohort for item in regressions)}
    valid = not failures
    quality = bool(entries and len(qualities) == count * 2 and all(qualities.values()))
    work_passed = not regressions
    advance = valid and complete and quality and work_passed
    executed = _work(entries)
    borrowed_native = {label: _native_work([state["traces"][label]["256"] for state in entries]) for label in ("old", "new")}
    borrowed_native["new"]["dual_reuse"] = _dual_work([state["traces"]["new"]["256"] for state in entries])
    oracles = [state["oracle"] for state in all_states if state["oracle"] is not None]
    return {"decision": "INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
            "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE" if advance else "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD",
        "state_count": count, "nonzero_state_count": counts["expected_nonzero_state_count"], "zero_state_count": count - counts["expected_nonzero_state_count"],
        "historical_state_count": sum(state["historical_regression"] for state in all_states),
        "failure_development_count": counts["expected_failure_development_count"], "axis_failure_development_count": counts["expected_axis_failure_development_count"],
        "executed_selector_calls": count * 2, "executed_interface_calls": count * 2, "native_search_constructions": count * 2,
        "executed_native_advance_calls": 0, "reference_native_search_constructions": 0,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows), "new_reference_oracle_calls": sum(row["new_reference_oracle_calls"] for row in rows),
        "borrowed_reference_count": count, "borrowed_native_snapshots": counts["expected_borrowed_native_snapshots"], "borrowed_prior_routed_count": count * 2,
        "execution_checks_passed": valid, "effective_references_complete": complete, "routed_quality_gate_passed": quality,
        "predecessor_work_gate_passed": work_passed, "work_gate_passed": work_passed, "development_gate_passed": advance,
        "failures": failures, "routed_nonclosures": nonclosures, "predecessor_work_regressions": regressions,
        "borrowed_native_score_comparisons": comparisons, "borrowed_native_score_regressions": [item for item in comparisons if item["score_regression"]],
        "all_executed_routed_work": executed, "borrowed_prior_routed_work": _prior_work(entries), "borrowed_native_work": borrowed_native,
        "executed_search_work": {key: sum(executed[label]["search_work"][key] for label in ("routed_old", "routed_new")) for key in COUNTERS},
        "executed_dual_reuse_work": executed["routed_new"]["dual_reuse"],
        "executed_axis_capacity_work": {label: executed[field]["axis_capacity"] for label, field in (("old", "routed_old"), ("new", "routed_new"))},
        "executed_frontier_export_work": executed["frontier_export_work"], "counterexample_states": counterexamples,
        "groups": groups, "cells": cells, "borrowed_reference_work": _combine_metrics(state["reference_metrics"] for state in all_states),
        "borrowed_reference_oracle_work": {"calls": len(oracles), "work": {key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"]}
