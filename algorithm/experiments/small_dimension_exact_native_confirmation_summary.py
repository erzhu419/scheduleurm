"""Independent small-LP confirmation with separate new and borrowed ledgers."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

from algorithm.experiments.affine_dual_development_summary import (
    _affine_work, _canonical_work, audit_affine_stage,
)
from algorithm.experiments.aligned_hardware_integer_capacity_development import COUNTERS, _combine_metrics
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import check_selection
from algorithm.experiments.branch_reserved_development_summary import audit_reserved_stage
from algorithm.experiments.direct_probe_own_tight_native_development_summary import _kernel_work as _direct_native_work
from algorithm.experiments.dual_reuse_development_summary import _dual_work
from algorithm.experiments.early_root_canonical_development_summary import _catalogue
from algorithm.experiments.integer_capacity_large_r_summary import _check_single_call
from algorithm.experiments.own_tight_dual_native_development_summary import _kernel_work as _own_native_work
from algorithm.experiments.root_pruning_development_summary import _native_work, _trace_errors
from algorithm.experiments.score_face_completion_confirmation_summary import ORACLE_COUNTERS, _reference
from algorithm.experiments.small_dimension_exact_native_development_summary import (
    SMALL_FIELDS, _audit_small_dimension_result, _kernel_work, _work,
)
from algorithm.experiments.strict_score_proof_timing_summary import _same
from algorithm.experiments.zero_delta_affine_development_summary import ZERO_COUNTERS, audit_zero_stage

SOURCE_FIELD = "small_dimension_confirmation_source"
FRESH_GROUP = "new_independent_fresh"
ARMS = (
    ("candidate", "routed"), ("direct", "previous_direct_probe_routed"),
    ("own", "previous_own_tight_routed"), ("fused", "previous_fused_root_routed"),
    ("settled", "previous_settled_root_routed"), ("native", "previous_native_root_routed"),
    ("face", "previous_dual_face_routed"), ("early", "previous_early_routed"),
    ("zero", "previous_zero_routed"), ("affine", "original_affine_routed"),
    ("branch", "old_routed"), ("long", "long_routed"),
)
LABELS = tuple(label for label, _ in ARMS)
ROOT_ARMS = dict(candidate="routed", own="previous_own_tight_routed", direct="previous_direct_probe_routed")


def _native_traces(state):
    return {
        "branch": state["traces"]["old"], "affine": state["original_affine_traces"],
        "zero": state["previous_zero_traces"], "own": state["previous_own_tight_native_traces"],
        "direct": state["previous_direct_probe_native_traces"], "candidate": state["candidate_native_traces"],
    }


def _native_ledger(states):
    finals = {label: [trace[label][str(state["caps"][-1])] for state, trace in
                      ((state, _native_traces(state)) for state in states)]
              for label in ("branch", "affine", "zero", "own", "direct", "candidate")}
    work = {}
    for label in ("branch", "affine", "zero"):
        stages = finals[label]
        work[label] = {**_native_work(stages), "canonical_work": _canonical_work([stage["selection"] for stage in stages])}
        if label != "branch":
            work[label].update(dual_reuse=_dual_work(stages), affine_dual=_affine_work(stages))
        if label == "zero":
            work[label]["zero_delta"] = {name: sum(stage["zero_delta"][name] for stage in stages) for name in ZERO_COUNTERS}
    work.update(own=_own_native_work(finals["own"]), direct=_direct_native_work(finals["direct"]),
                candidate=_kernel_work(finals["candidate"]))
    return work


def _paid_work(states, fresh):
    return {
        "whole_selector": {label: _work([state[field] for state in states], label)
                           for label, field in (ARMS if fresh else ARMS[:-1])},
        "native": _native_ledger(states),
    }


def _group(states):
    fresh = [state for state in states if state[SOURCE_FIELD] == "fresh"]
    borrowed = [state for state in states if state[SOURCE_FIELD] == "borrowed_control"]
    return {"state_count": len(states), "independent_state_count": len(fresh), "borrowed_state_count": len(borrowed),
            "executed": _paid_work(fresh, True), "borrowed": _paid_work(borrowed, False)}


def _native_errors(state, protocol, model, lanes, expected, reference):
    """Six actual ladders; shared-kernel trace aliases are borrowed only once."""
    options = tuple(SimpleNamespace(**option) for option in model["options"])
    _, thresholds = _catalogue(model)
    roots = len(thresholds)
    caps = (roots, roots + 1, protocol["initial_lp_budget"], protocol["lp_node_budget"])
    traces = _native_traces(state)
    if tuple(state["caps"]) != caps or any(set(trace) != {str(cap) for cap in caps} for trace in traces.values()):
        return ["every genuine native engine must retain all four registered caps"]
    errors = []
    for field in ("previous_dual_face_traces", "previous_native_root_traces", "previous_settled_root_traces", "previous_fused_root_traces"):
        if not _same(state[field], traces["zero"]):
            errors.append(field + " must retain its genuine shared zero-engine ladder")
    if not _same(state["traces"]["new"], traces["zero"]):
        errors.append("traces.new must retain the actual zero-engine ladder")
    for label in ("branch", "affine", "zero"):
        trace, previous = traces[label], None
        errors.extend(label + ": " + message for message in _trace_errors(
            trace, state, options, lanes, expected, reference, True))
        for cap in caps:
            stage = trace[str(cap)]
            if not _same(stage["thresholds"], thresholds):
                errors.append(label + " duration strata differ from the original catalogue")
            errors.extend(label + f" cap{cap}: " + message for message in audit_reserved_stage(stage, cap, roots))
            if label != "branch":
                errors.extend(label + f" cap{cap}: " + message for message in audit_affine_stage(stage, len(state["queue"]), previous))
            if label == "zero":
                errors.extend(label + f" cap{cap}: " + message for message in audit_zero_stage(stage, previous, traces["affine"][str(cap)]))
            previous = stage
    # The candidate/own/direct whole auditors below independently validate
    # their raw, settled, four cumulative prefixes and separate copy ledgers.
    return errors


def _oracle_work(states):
    oracles = [state["oracle"] for state in states if state["oracle"] is not None]
    return {"calls": len(oracles), "work": {name: sum(oracle["work"][name] for oracle in oracles) for name in ORACLE_COUNTERS}}


def summarize(protocol, rows):
    if [(row["node"], row["lanes"]) for row in rows] != [(cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        raise ValueError("all registered confirmation cells must occur once in order")
    models = {model["node"]: model for model in protocol["models"]}
    failures, nonclosures, entries, groups = [], [], [], defaultdict(list)
    complete = True
    for cell, row in zip(protocol["cells"], rows, strict=True):
        model, states = models[cell["node"]], row["states"]
        if len(states) != len(cell["states"]) or not states or any(not _same(
                {name: state.get(name) for name in original}, original)
                for state, original in zip(states, cell["states"], strict=True)):
            raise ValueError("every new queue and every borrowed original metadata/source pointer must remain frozen")
        fresh_states = [state for state in states if state[SOURCE_FIELD] == "fresh"]
        fresh, point = len(fresh_states), {"node": row["node"], "lanes": row["lanes"]}
        phases = {label: sum(state[field]["root_settle_work"] is not None for state in fresh_states)
                  for label, field in ROOT_ARMS.items()}
        options = tuple(SimpleNamespace(**option) for option in model["options"])
        if (row["registered_at_utc"] != protocol["registered_at_utc"] or row["physical_lanes"] != model["lanes"]
                or row["option_count"] != len(options) or row["new_reference_calls"] != fresh
                or row["new_oracle_calls"] != sum(state["oracle"] is not None for state in fresh_states)
                or row["selector_calls"] != dict.fromkeys(LABELS, fresh)
                or row["native_advance_calls"] != 24 * fresh or row["native_search_constructions"] != 18 * fresh
                or row["root_settle_calls"] != phases or any(
                    row[name]["cpu_affinity"] != [protocol["cpu"]] or not row[name]["gc_enabled"]
                    for name in ("runtime_start", "runtime_end"))):
            failures.append({**point, "errors": ["runtime, catalogue or actual fresh reference/whole/native/phase coverage differs"]})
        for state in states:
            pointer = {**point, **{name: state[name] for name in ("state_id", "queue", "timing_group", SOURCE_FIELD)}}
            source, errors = state[SOURCE_FIELD], []
            if source not in ("fresh", "borrowed_control"):
                raise ValueError("confirmation source must identify an actually new or borrowed record")
            if (source == "fresh") != (state["timing_group"] == FRESH_GROUP):
                errors.append("new confirmation queues must retain their own cohort, separate from every old cohort")
            if source == "fresh" and (state["confirmation_source"] != "fresh" or "prior_routed" in state):
                errors.append("new queues require fresh audits without an invented axis predecessor")
            if source == "borrowed_control" and "long_routed" in state:
                errors.append("borrowed development records cannot invent an unexecuted long result")
            if source == "borrowed_control":
                # The predecessor's complete quality pass owns these exact
                # records. Preserve their metadata and aggregate borrowed fees.
                complete &= bool(state["effective_reference_complete"])
                if errors:
                    failures.append({**pointer, "errors": errors})
                entries.append(state)
                groups[state["timing_group"]].append(state)
                continue
            reference, expected = selection_from_record(state["reference"]), None
            try:
                _check_single_call(state["reference_metrics"], state["reference"], reference_complete=state["reference_complete"])
                expected, extra = _reference(state, protocol, len(options))
                errors.extend(extra)
            except ValueError as error:
                errors.append(str(error))
            metric = state["reference_metrics"]
            if (state["reference"]["lp_node_budget"] != protocol["reference_budget"] or metric["failures"]
                    or metric["reference_comparable_count"] != int(state["reference_complete"])
                    or any(metric["search_work"][name] != state["reference"][name] for name in COUNTERS)):
                errors.append("independent or borrowed reference has invalid budget, certificate or recorded work")
            errors.extend("reference: " + name for name, failed in check_selection(reference, options,
                tuple(state["queue"]), row["lanes"], protocol["reference_budget"], expected).items() if failed)
            complete &= expected is not None
            errors.extend(_native_errors(state, protocol, model, row["lanes"], expected, reference))
            for label, field in ARMS:
                try:
                    _, extra, quality = _audit_small_dimension_result(state[field], state, protocol, model, row["lanes"], label, expected)
                except ValueError as error:
                    extra, quality = [str(error)], False
                errors.extend(label + ": " + message for message in extra)
                if not quality:
                    result = state[field]
                    status = ({"status": result["completion"]["status"], "score_proof_status": result["score_proof"]["status"]}
                              if label == "long" else {"status": result["status"]})
                    nonclosures.append({**pointer, "label": label, **status})
            if errors:
                failures.append({**pointer, "errors": errors})
            entries.append(state)
            groups[state["timing_group"]].append(state)
    fresh_states = [state for state in entries if state[SOURCE_FIELD] == "fresh"]
    borrowed = [state for state in entries if state[SOURCE_FIELD] == "borrowed_control"]
    fresh = len(fresh_states)
    counts = dict(expected_state_count=len(entries), expected_independent_state_count=fresh,
        expected_borrowed_state_count=len(borrowed), expected_selector_calls=12 * fresh,
        expected_native_advance_calls=24 * fresh, expected_native_constructions=18 * fresh,
        expected_interface_calls=36 * fresh, expected_new_reference_calls=fresh,
        expected_candidate_native_snapshots=4 * fresh)
    if any(protocol[name] != count for name, count in counts.items()) or protocol["expected_group_counts"] != dict(Counter(
            state["timing_group"] for state in entries)):
        failures.append({"errors": ["complete fresh/borrowed cohort or registered method/reference accounting differs"]})
    executed, borrowed_work = _paid_work(fresh_states, True), _paid_work(borrowed, False)
    execution = not failures
    advance = bool(execution and complete and not nonclosures)
    candidate = executed["whole_selector"]["candidate"]
    return {
        "schema_version": 1, "experiment": protocol["experiment"], "registered_at_utc": protocol["registered_at_utc"],
        "decision": "INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not complete else
            "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_ADVANCE" if advance else "SMALL_DIMENSION_EXACT_NATIVE_CONFIRMATION_HOLD",
        "state_count": len(entries), "independent_state_count": fresh, "borrowed_state_count": len(borrowed),
        "independent_confirmation": True, "execution_checks_passed": execution, "references_complete": bool(complete),
        "quality_gate_passed": advance, "work_gate_passed": execution, "confirmation_gate_passed": advance,
        "failures": failures, "nonclosures": nonclosures,
        "new_reference_calls": sum(row["new_reference_calls"] for row in rows),
        "new_oracle_calls": sum(row["new_oracle_calls"] for row in rows), "borrowed_reference_count": len(borrowed),
        "executed_selector_calls": sum(sum(row["selector_calls"].values()) for row in rows),
        "executed_selector_calls_by_label": {label: sum(row["selector_calls"][label] for row in rows) for label in LABELS},
        "executed_native_advance_calls": sum(row["native_advance_calls"] for row in rows),
        "executed_interface_calls": sum(sum(row["selector_calls"].values()) + row["native_advance_calls"] for row in rows),
        "native_search_constructions": sum(row["native_search_constructions"] for row in rows),
        "executed_native_snapshots": 24 * fresh, "executed_candidate_native_snapshots": 4 * fresh,
        "executed_root_settle_calls": {label: sum(row["root_settle_calls"][label] for row in rows) for label in ROOT_ARMS},
        "executed_small_dimension_lp_work": candidate[SMALL_FIELDS[0]],
        "executed_small_dimension_lp_diagnostics": candidate["small_dimension_lp_diagnostics"],
        "executed": executed, "borrowed": borrowed_work, "groups": {name: _group(states) for name, states in groups.items()},
        "reference_work": {"new": _combine_metrics(state["reference_metrics"] for state in fresh_states),
                           "borrowed": _combine_metrics(state["reference_metrics"] for state in borrowed)},
        "oracle_work": {"new": _oracle_work(fresh_states), "borrowed": _oracle_work(borrowed)},
        "prior_decisions": protocol["prior_decisions"], "prior_hold_retained": True, "limitations": protocol["limitations"],
    }
