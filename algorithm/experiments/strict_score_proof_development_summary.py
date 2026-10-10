"""Audit shared-budget score proof development and retained history separately."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json

from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.score_face_completion_confirmation_summary import (
    LABELS, ORACLE_COUNTERS, _aggregate as _completion_aggregate, _completion_errors,
    _key, _pointer, _reference, _score, _selection_equal,
)


PROOF_COUNTERS = (
    "states_visited", "memo_hits", "count_branches", "bound_prunes", "duration_prunes",
    "gcd_prunes", "failed_states_cached", "thresholds_checked",
    "suffix_coordinate_reads", "resource_coordinate_reads",
)


def _same(left, right):
    return json.loads(json.dumps(left)) == json.loads(json.dumps(right))


def _witness_valid(witness, model, queue, lanes, baseline_score):
    options, counts = model["options"], witness["option_counts"]
    if len(counts) != len(options) or any(type(n) is not int or n < 0 for n in counts) or sum(counts) > lanes:
        return False
    service = tuple(sum(n * option["requirements"][j] for n, option in zip(counts, options)) for j in range(len(queue)))
    duration = max((1.0, *(option["duration"] for n, option in zip(counts, options) if n)))
    indices = tuple(i for i, n in enumerate(counts) for _ in range(n)) + (None,) * (lanes - sum(counts))
    score = Fraction(sum(q * s for q, s in zip(queue, service))) / Fraction.from_float(duration)
    return (all(s <= q for s, q in zip(service, queue)) and tuple(witness["service"]) == service
            and tuple(witness["option_indices"]) == indices and witness["duration"] == duration
            and Fraction(*witness["score_fraction"]) == score and score > baseline_score)


def _proof_errors(state, protocol, model, lanes):
    record, baseline = state["completed"], state["baseline"]["selection"]
    proof, intermediate = record["score_proof"], record["score_certified_selection"]
    cap, errors = protocol["integer_state_budget"], []
    if record["integer_state_budget"] != cap or proof["state_budget"] != cap:
        raise ValueError("proof state budget differs from the registered shared cap")
    if not _selection_equal(baseline, record["baseline_selection"]):
        errors.append("embedded stage-one snapshots disagree")
    if any(type(proof[key]) is not int or proof[key] < 0 for key in PROOF_COUNTERS):
        errors.append("proof work counters must be nonnegative integers")
    if proof["states_visited"] + record["completion"]["states_visited"] > cap:
        errors.append("proof and completion exceeded their shared integer-state cap")
    attempted = not baseline["score_exact"] and not baseline["canonical_exact"]
    if proof["attempted"] != attempted:
        errors.append("strict-proof trigger disagrees with the original score certificate")
    expected_intermediate = dict(baseline)
    if not attempted:
        status = "SKIPPED_CANONICAL_COMPLETE" if baseline["canonical_exact"] else "SKIPPED_SCORE_CERTIFIED"
        if (proof["status"] != status or not proof["complete"] or proof["score_fraction"] is not None
                or proof["witness"] is not None or any(proof[key] for key in PROOF_COUNTERS)):
            errors.append("skipped proof must preserve the existing certificate with zero work")
    else:
        score = _score(baseline, state["queue"])
        if proof["score_fraction"] is None or tuple(proof["score_fraction"]) != (score.numerator, score.denominator):
            errors.append("strict-proof target differs from its own baseline score")
        if proof["status"] == "COMPLETE":
            if not proof["complete"] or proof["witness"] is not None:
                errors.append("successful absence proof has inconsistent completion or witness")
            expected_intermediate.update(score_exact=True, upper_bound=baseline["score"], gap=0.0, epsilon=0.0)
        elif proof["status"] == "BETTER_ACTION_FOUND":
            if proof["complete"] or proof["witness"] is None or not _witness_valid(proof["witness"], model, state["queue"], lanes, score):
                errors.append("strict-improvement witness is not a feasible strictly better action")
        elif proof["status"] == "STATE_BUDGET_EXHAUSTED":
            if proof["complete"] or proof["witness"] is not None or proof["states_visited"] != cap:
                errors.append("exhausted proof has inconsistent certificate or charged work")
        else:
            errors.append("attempted strict proof has an unsupported outcome")
    if not _selection_equal(intermediate, expected_intermediate):
        errors.append("intermediate score certificate changed unsupported baseline fields")
    # Audit canonical completion against its actual input, the intermediate
    # certificate, rather than pretending the original LP stage proved the score.
    view = {**state, "baseline_source": "embedded_stage_one",
            "baseline": {**state["baseline"], "selection": intermediate},
            "completed": {**record, "baseline_selection": intermediate}}
    errors.extend(_completion_errors(view, {"canonical_state_budget": cap - proof["states_visited"]}))
    return errors


def _history_errors(state):
    previous, reused = state["previous"], state["reference_reused"]
    if reused != (state["source"] != "new_neighborhood") or reused != (previous is not None):
        raise ValueError("reference reuse and previous snapshot differ from the registered source")
    if not reused:
        if state["baseline_reproduced"] is not None or state["completion_reproduced"] is not None:
            raise ValueError("new neighbors have no historical reproduction comparison")
        return []
    record = state["completed"]
    baseline_equal = (_selection_equal(record["baseline_selection"], previous["baseline_selection"])
                      and record["metrics"]["preprocessing"] == previous["preprocessing"]
                      and record["capacity_propagation"] == previous["capacity_propagation"])
    errors = []
    if not baseline_equal or state["baseline_reproduced"] != baseline_equal:
        errors.append("stage one did not reproduce its frozen historical baseline")
    required = previous["baseline_selection"]["score_exact"]
    completed_equal = (_selection_equal(record["selection"], previous["selection"])
                       and _same(record["completion"], previous["completion"])) if required else None
    if (required and not completed_equal) or state["completion_reproduced"] != completed_equal:
        errors.append("already score-certified path did not reproduce frozen completion")
    return errors


def _reference_accounting(entries):
    oracles = [state["oracle"] for _, state in entries if state["oracle"] is not None]
    return {"calls": len(entries), "metrics": _combine_metrics(state["reference_metrics"] for _, state in entries),
            "oracle_calls": len(oracles), "oracle_work": {
                key: sum(oracle["work"][key] for oracle in oracles) for key in ORACLE_COUNTERS}}


def _aggregate(entries):
    result = _completion_aggregate(entries)
    proofs = [state["completed"]["score_proof"] for _, state in entries]
    shared = [proof["states_visited"] + state["completed"]["completion"]["states_visited"]
              for proof, (_, state) in zip(proofs, entries)]
    result.update(score_proof={
        "attempted_count": sum(proof["attempted"] for proof in proofs),
        "completed_count": sum(proof["status"] == "COMPLETE" for proof in proofs),
        "better_action_found_count": sum(proof["status"] == "BETTER_ACTION_FOUND" for proof in proofs),
        "exhausted_count": sum(proof["status"] == "STATE_BUDGET_EXHAUSTED" for proof in proofs),
        "work": {key: sum(proof[key] for proof in proofs) for key in PROOF_COUNTERS},
        "max_states_visited": max((proof["states_visited"] for proof in proofs), default=0)},
        shared_integer_states=sum(shared), max_shared_integer_states=max(shared, default=0),
        baseline_reproduction_required_count=sum(state["previous"] is not None for _, state in entries),
        baseline_reproduced_count=sum(state["baseline_reproduced"] is True for _, state in entries),
        completion_reproduction_required_count=sum(state["previous"] is not None and state["previous"]["baseline_selection"]["score_exact"] for _, state in entries),
        completion_reproduced_count=sum(state["completion_reproduced"] is True for _, state in entries),
        proof_trigger_points=[{**_pointer(row, state), "score_proof": state["completed"]["score_proof"],
                               "completion": state["completed"]["completion"],
                               "final_score_exact": state["completed"]["selection"]["score_exact"],
                               "final_canonical_exact": state["completed"]["selection"]["canonical_exact"]}
                              for row, state in entries if state["completed"]["score_proof"]["attempted"]])
    return result


def summarize(protocol, rows):
    models = {model["node"]: model for model in protocol["models"]}
    cells_expected = {(node, lanes) for node in models for lanes in protocol["lanes"]}
    cells_actual = [(row["node"], row["lanes"]) for row in rows]
    if len(cells_actual) != len(set(cells_actual)) or set(cells_actual) != cells_expected:
        raise ValueError("all registered cells must occur exactly once")
    entries, cells, cohorts, lanes_groups, failures = [], [], defaultdict(list), defaultdict(list), []
    for row in rows:
        model = models[row["node"]]
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(model["options"])):
            raise ValueError("cell registration or catalogue metadata differs from the protocol")
        panels = protocol["queue_panels"][str(row["lanes"]) ]
        expected = [(state["state_id"], panel["panel"], tuple(state["queue"]), panel["cohort"], state["source"], state["source_state_id"])
                    for panel in panels for state in panel["states"]]
        actual = [(state["state_id"], state["panel"], tuple(state["queue"]), state["cohort"], state["source"], state["source_state_id"]) for state in row["states"]]
        if actual != expected or not actual or len({x[0] for x in actual}) != len(actual) or len({x[2] for x in actual}) != len(actual):
            raise ValueError("stored states differ from the registered deduplicated roster and sources")
        cell_entries = []
        for state in row["states"]:
            if state["source"] != "new_neighborhood" and state["source"] not in protocol["prior_sources"]:
                raise ValueError("unknown registered historical source")
            _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"]
                    or state["reference_metrics"]["reference_comparable_count"] != int(state["reference_complete"])):
                raise ValueError("reference budget or original reference comparability differs from protocol")
            expected_key, errors = _reference(state, protocol, row["option_count"])
            records = [(label, state[label]["metrics"], state[label]["selection"]) for label in LABELS]
            records.append(("score_certified", state["score_certified_metrics"], state["completed"]["score_certified_selection"]))
            for label, metrics, selected in records:
                _check_single_call(metrics, selected)
                if selected["lp_node_budget"] != protocol["budget"] or metrics["reference_comparable_count"] != int(expected_key is not None):
                    raise ValueError("candidate budget or reference comparability differs from protocol")
                if (selected["lp_calls"] > protocol["budget"] or metrics["failures"]
                        or any(metrics["search_work"][key] != selected[key] for key in COUNTERS)):
                    errors.append(f"{label} has invalid selection metrics or LP accounting")
                if expected_key is not None:
                    key = _key(selected, state["queue"], row["option_count"])
                    if (metrics["score_mismatch_count"] != int(key[0] != expected_key[0])
                            or metrics["canonical_action_mismatch_count"] != int(key[1:] != expected_key[1:])):
                        errors.append(f"{label} reference comparison disagrees with its exact action")
            errors.extend(_proof_errors(state, protocol, model, row["lanes"]))
            errors.extend(_history_errors(state))
            if state["reference_metrics"]["failures"]:
                errors.append("full reference has invalid selection metrics")
            if errors:
                failures.append({**_pointer(row, state), "errors": errors})
            entry = (row, state)
            entries.append(entry)
            cell_entries.append(entry)
            cohorts[state["cohort"]].append(entry)
            lanes_groups[row["lanes"]].append(entry)
        cells.append({"node": row["node"], "lanes": row["lanes"], **_aggregate(cell_entries)})
    total = _aggregate(entries)
    valid, complete = not failures, not total["effective_incomplete_reference_count"]
    action = bool(valid and complete and total["action_gate_passed"])
    decision = ("INVALID_EXECUTION" if not valid else "INCOMPLETE_REFERENCE" if not complete else
                "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE" if action else "STRICT_SCORE_PROOF_DEVELOPMENT_HOLD")
    executed = _reference_accounting([entry for entry in entries if not entry[1]["reference_reused"]])
    reused = _reference_accounting([entry for entry in entries if entry[1]["reference_reused"]])
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": decision, "budget": protocol["budget"], "integer_state_budget": protocol["integer_state_budget"],
        "cell_count": len(rows), "executed_selector_calls": len(entries) + executed["calls"],
        "executed_candidate_calls": len(entries), "embedded_baseline_count": len(entries),
        "executed_reference_calls": executed["calls"], "reused_reference_calls": reused["calls"],
        "executed_oracle_calls": executed["oracle_calls"], "reused_oracle_calls": reused["oracle_calls"],
        "certificate_checks_passed": valid, "proof_accounting_failures": failures,
        "original_references_complete": not total["original_incomplete_reference_count"],
        "effective_references_complete": complete, "development_action_gate_passed": action, **total,
        "executed_reference": executed, "reused_reference": reused,
        "executed_search_work": {key: executed["metrics"]["search_work"][key] + total["metrics"]["completed"]["search_work"][key] for key in COUNTERS},
        "cells": cells, "cohorts": {name: _aggregate(group) for name, group in cohorts.items()},
        "by_lanes": [{"lanes": lanes, **_aggregate(group)} for lanes, group in lanes_groups.items()],
        "limitations": protocol["limitations"],
    }
