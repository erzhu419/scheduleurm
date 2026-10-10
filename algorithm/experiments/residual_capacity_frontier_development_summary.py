"""Audit retained capacity tables and budgeted canonical prefixes on reused states."""
from __future__ import annotations

from fractions import Fraction

from algorithm.experiments.axis_capacity_frontier_development_summary import summarize as _axis_summary, _work as _axis_work
from algorithm.experiments.strict_score_proof_timing_summary import _same

RESIDUAL_COUNTERS = ("prefix_states", "dfs_states_visited", "prefix_reads", "capacity_queries",
                     "capacity_prunes", "tables_reused", "coverage_coordinate_reads")


def _merged_values(profiles, queue, lanes):
    merged = [0] + [-1] * lanes
    for waiting, profile in zip(queue, profiles):
        merged = [max((merged[slots - used] + waiting * profile[used]
            for used in range(slots + 1) if merged[slots - used] >= 0 and profile[used] >= 0), default=-1)
            for slots in range(lanes + 1)]
    return merged


def _original_proof(proof):
    capacity = proof["capacity"]
    return {**proof, "capacity": {**{key: value for key, value in capacity.items() if key != "merged_values_copied"},
        "bounds": [{key: value for key, value in bound.items() if key != "exact_slot_numerators"} for bound in capacity["bounds"]]}}


def _coverage(bounds, thresholds, requirements, durations, score):
    reads = used = 0
    for threshold in thresholds:
        if (score * threshold).denominator != 1:
            continue
        for bound in bounds:
            if Fraction(*bound["threshold_fraction"]) != threshold:
                continue
            represented = set(map(tuple, bound["requirements"]))
            reads += sum(map(len, bound["requirements"]))
            for required, duration in zip(requirements, durations):
                if duration > threshold:
                    continue
                reads += len(required)
                if any(required) and required not in represented:
                    break
            else:
                used += 1
                break
    return reads, used


def audit_residual_completion(probe, prior_probe, model, queue, lanes):
    errors = []
    proof, work = probe["score_proof"], probe["completion"]
    bounds = proof["capacity"]["bounds"]
    baseline = probe["score_certified_selection"]
    score = Fraction(sum(q * done for q, done in zip(queue, baseline["service"]))) / Fraction.from_float(baseline["duration"])
    durations = [Fraction.from_float(float(max(1., option["duration"]))) for option in model["options"]]
    requirements = [tuple(option["requirements"]) for option in model["options"]]
    thresholds = sorted({duration for duration in durations if duration <= Fraction.from_float(baseline["duration"])})
    relevant = bool(proof["complete"] and score and any(Fraction(*bound["threshold_fraction"]) in thresholds
                    and (score * Fraction(*bound["threshold_fraction"])).denominator == 1 for bound in bounds))
    if not relevant:
        if not _same(work, prior_probe["completion"]) or not _same(probe["selection"], prior_probe["selection"]):
            errors.append("without a relevant retained table, canonical selection and all previous work must remain identical")
        return errors
    if any(key not in work for key in RESIDUAL_COUNTERS):
        return ["canonical completion must record its prefix, DFS and coverage work"]
    if any(type(work[key]) is not int or work[key] < 0 for key in RESIDUAL_COUNTERS):
        errors.append("residual canonical work counters must be nonnegative integers")
    if work["states_visited"] != work["prefix_states"] + work["dfs_states_visited"] or work["prefix_reads"] != work["prefix_states"]:
        errors.append("canonical visited states must charge each prefix read and every actual DFS state")
    if work["states_visited"] > work["state_budget"] or work["state_budget"] != probe["integer_state_budget"] - proof["states_visited"]:
        errors.append("prefix and canonical DFS must use only the unchanged shared proof remainder")
    if work["thresholds_checked"] > len(thresholds):
        errors.append("canonical threshold checks exceed the original duration roster")
    reads, used = _coverage(bounds, thresholds[:work["thresholds_checked"]], requirements, durations, score)
    if reads != work["coverage_coordinate_reads"] or used != work["tables_reused"]:
        errors.append("table reuse must cover all original allowed requirements and charge every inspected coordinate")
    full = work["tables_reused"] * (lanes + 1)
    if (work["prefix_states"] > full or work["prefix_states"] < max(0, full - lanes - 1)
            or work["prefix_states"] != full and work["status"] != "STATE_BUDGET_EXHAUSTED"):
        errors.append("prefix work must include every selected table cell except a final budget-exhausted prefix")
    if (work["capacity_prunes"] > work["capacity_queries"] or work["capacity_queries"] > work["dfs_states_visited"]
            or work["capacity_prunes"] > work["bound_prunes"]
            or not work["tables_reused"] and (work["capacity_queries"] or work["capacity_prunes"])):
        errors.append("residual queries and strict capacity prunes must be actual covered canonical DFS operations")
    if not work["tables_reused"]:
        original = {key: value for key, value in work.items() if key not in RESIDUAL_COUNTERS}
        if not _same(original, prior_probe["completion"]) or not _same(probe["selection"], prior_probe["selection"]):
            errors.append("an unused or coverage-rejected table must preserve the previous canonical DFS and selection")
    return errors


def _work(states):
    work = _axis_work(states)
    for label, field in (("old", "routed_old"), ("new", "routed_new")):
        record_field = "old_routed" if label == "old" else "routed"
        probes = [state[record_field]["probe"] for state in states if state[record_field]["probe"] is not None]
        work[field]["axis_capacity"]["merged_values_copied"] = sum(probe["score_proof"].get("capacity", {}).get("merged_values_copied", 0) for probe in probes)
        work[field]["residual_capacity_completion"] = {"extended_calls": sum("prefix_states" in probe["completion"] for probe in probes),
            **{key: sum(probe["completion"].get(key, 0) for probe in probes) for key in RESIDUAL_COUNTERS}}
        work[field]["canonical_dfs_states"] = sum(probe["completion"].get("dfs_states_visited", probe["completion"]["states_visited"]) for probe in probes)
    return work


def summarize(protocol, rows):
    result = _axis_summary(protocol, rows)
    models = {model["node"]: model for model in protocol["models"]}
    failures = []
    states = [state for row in rows for state in row["states"]]
    by_identity = {}
    for row in rows:
        model = models[row["node"]]
        for state in row["states"]:
            point = {"node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"], "queue": state["queue"], "cohort": state["cohort"]}
            by_identity[row["node"], row["lanes"], state["state_id"]] = state
            for label, field in (("old", "old_routed"), ("new", "routed")):
                current, prior = state[field], state["prior_routed"][label]
                if prior["probe_stage"] != "frontier_late":
                    continue  # The frozen axis summary already audits ordinary full-output equality.
                if current["probe"] is None:
                    continue  # Missing frontier probes are already execution failures.
                probe, prior_probe = current["probe"], prior["probe"]
                proof, errors = probe["score_proof"], []
                if not _same(_original_proof(proof), prior_probe["score_proof"]):
                    errors.append("retaining a table must preserve every original axis/DFS proof field and arithmetic counter")
                capacity = proof["capacity"]
                if capacity.get("merged_values_copied") != len(capacity["bounds"]) * (row["lanes"] + 1):
                    errors.append("every completed table must charge all retained merged-vector values exactly once")
                for bound in capacity["bounds"]:
                    if not _same(bound.get("exact_slot_numerators"), _merged_values(bound["axis_service_profiles"], state["queue"], row["lanes"])):
                        errors.append("retained exact-slot numerators differ from the already-paid max-plus merge")
                errors.extend(audit_residual_completion(probe, prior_probe, model, state["queue"], row["lanes"]))
                if errors:
                    failures.append({**point, "label": label, "errors": errors})
    result["failures"].extend(failures)
    valid = not result["failures"]
    advance = valid and result["effective_references_complete"] and result["routed_quality_gate_passed"] and result["predecessor_work_gate_passed"]
    work = _work(states)
    result.update(decision="INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not result["effective_references_complete"] else
        "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE" if advance else "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_HOLD",
        execution_checks_passed=valid, development_gate_passed=advance, retained_capacity_checks_passed=not failures,
        all_executed_routed_work=work,
        executed_axis_capacity_work={label: work[field]["axis_capacity"] for label, field in (("old", "routed_old"), ("new", "routed_new"))},
        executed_residual_completion_work={label: work[field]["residual_capacity_completion"] for label, field in (("old", "routed_old"), ("new", "routed_new"))})
    for cohort, group in result["groups"].items():
        group["work"] = _work([state for state in states if state["cohort"] == cohort])
    for cell in result["cells"]:
        cell["work"] = _work([state for row in rows if (row["node"], row["lanes"]) == (cell["node"], cell["lanes"]) for state in row["states"]])
    for example in result["counterexample_states"]:
        example["work"] = _work([by_identity[example["node"], example["lanes"], example["state_id"]]])
    return result
