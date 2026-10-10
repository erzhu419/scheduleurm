"""Check native root checkpoints against the complete frozen dual-face path."""
from __future__ import annotations

from algorithm.experiments.dual_face_root_canonical_development_summary import (
    DUAL_FACE_COUNTERS, ROUTING_COUNTERS, _audit_dual_face_result,
    _work as _face_work, summarize as _face_summary,
)
from algorithm.experiments.resumable_probe_development_summary import _exact, _metric
from algorithm.experiments.strict_score_proof_timing_summary import _same

REMOVED_ROUTING_COUNTERS = ROUTING_COUNTERS[:8]
SNAPSHOT_COUNTERS = ROUTING_COUNTERS[8:]


def _path_errors(result, state):
    previous = state["previous_dual_face_routed"]
    work = result.get("routing_preparation", {})
    errors = []
    if (set(work) != set(ROUTING_COUNTERS)
            or any(type(value) is not int or value < 0 for value in work.values())
            or any(work.get(name) != 0 for name in REMOVED_ROUTING_COUNTERS)):
        errors.append("the eight eliminated root-grouping operations must have actual zero fees")
    if any(work.get(name) != previous["routing_preparation"][name] for name in SNAPSHOT_COUNTERS):
        errors.append("native root checkpoints must retain all three actual root snapshot fees")
    if not _same({name: value for name, value in result.items() if name != "routing_preparation"},
                 {name: value for name, value in previous.items() if name != "routing_preparation"}):
        errors.append("native root action, raw snapshots, LP/probe path, exact dual evidence or paid work differs from the frozen parent")
    traces = state["previous_dual_face_traces"]
    if (not _same(traces, state["previous_zero_traces"])
            or not _same(traces, state["traces"]["new"])
            or set(traces) != {str(cap) for cap in state["caps"]}):
        errors.append("the genuine four-cap parent native ladder must remain complete and unchanged")
    return errors


def _audit_native_root_result(result, state, protocol, model, lanes, label, expected):
    if label != "candidate":
        metric, errors, quality = _audit_dual_face_result(result, state, protocol, model, lanes,
            "candidate" if label == "face" else label, expected)
        if label == "face" and not _same(result, state["previous_dual_face_routed"]):
            errors.append("face control differs from its actual frozen whole-selector result")
        return metric, errors, quality
    # Audit the real saved parent, retaining its real paid preparation work.
    # Candidate equality below independently checks every other field; its
    # deleted fees remain zero throughout the executed-work aggregation.
    _, errors, _ = _audit_dual_face_result(state["previous_dual_face_routed"], state,
        protocol, model, lanes, "candidate", expected)
    errors.extend(_path_errors(result, state))
    budget = state["previous_dual_face_routed"]["final_lp"]["selection"]["lp_node_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, result["final_lp"]["preprocessing"])
    if metric["failures"]:
        errors.append("native checkpoint returned an invalid action or certificate")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(results, label):
    return _face_work(results, "candidate" if label == "face" else label)


def summarize(protocol, rows):
    # The frozen summary validates the genuine borrowed parent records, rather
    # than substituting old preparation fees into the new candidate records.
    parent_rows = [{**row, "states": [{**state, "routed": state["previous_dual_face_routed"]}
                                     for state in row["states"]]} for row in rows]
    result = _face_summary(protocol, parent_rows)
    states = [state for row in rows for state in row["states"]]
    path_failures = []
    for row in rows:
        for state in row["states"]:
            errors = _path_errors(state["routed"], state)
            if errors:
                path_failures.append({"node": row["node"], "lanes": row["lanes"],
                    **{key: state[key] for key in ("state_id", "queue", "timing_group")}, "errors": errors})
    result["failures"].extend(path_failures)
    executed = _work([state["routed"] for state in states], "candidate")
    result["executed"]["whole_selector"] = executed
    result["executed_routing_preparation_work"] = executed["routing_preparation"]
    result["executed_dual_face_work"] = executed["dual_face_work"]
    result["borrowed"]["whole_selector"]["face"] = _work([state["previous_dual_face_routed"] for state in states], "face")
    result["removed_routing_preparation_work"] = {name: sum(state["previous_dual_face_routed"]["routing_preparation"][name]
        for state in states) for name in REMOVED_ROUTING_COUNTERS}
    result["borrowed_dual_face_native_snapshots"] = sum(len(state["previous_dual_face_traces"]) for state in states)
    result["behavior_preserved"] = not path_failures
    for name, group in result["groups"].items():
        items = [state for state in states if state["timing_group"] == name]
        group["executed"]["whole_selector"] = _work([state["routed"] for state in items], "candidate")
        group["borrowed"]["whole_selector"]["face"] = _work([state["previous_dual_face_routed"] for state in items], "face")
    result["root_probe_states"] = [{"node": row["node"], "lanes": row["lanes"],
        **{name: state[name] for name in ("state_id", "queue")},
        "completion_status": state["routed"]["probe"]["completion"]["status"],
        "work": _work([state["routed"]], "candidate")} for row in rows for state in row["states"]
        if state["routed"]["probe_stage"] == "root"]
    execution = not result["failures"]
    advance = execution and result["references_complete"] and not result["nonclosures"]
    result.update(execution_checks_passed=execution, work_gate_passed=execution,
        quality_gate_passed=bool(advance), development_gate_passed=bool(advance),
        decision="INVALID_EXECUTION" if not execution else "INCOMPLETE_REFERENCE" if not result["references_complete"] else
            "NATIVE_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE" if advance else "NATIVE_ROOT_DUAL_FACE_DEVELOPMENT_HOLD")
    return result
