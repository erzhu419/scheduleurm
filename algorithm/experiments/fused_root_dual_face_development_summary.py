"""Preserve the settled algorithm while charging its single-pass checkpoints."""
from __future__ import annotations

from algorithm.experiments.resumable_probe_development_summary import _exact, _metric
from algorithm.experiments.settled_root_dual_face_checked_development_summary import (
    DUAL_FACE_COUNTERS, ROOT_SETTLE_COUNTERS, ROOT_SETTLE_FEE_COUNTERS, ROUTING_COUNTERS,
    _audit_settled_root_result, _work as _settled_work, summarize as _settled_summary,
)
from algorithm.experiments.strict_score_proof_timing_summary import _same

ROOT_SCAN_FIELDS = ("root_snapshot_heap_entries_read", "root_snapshot_retained_entries_read")
SETTLED_SCAN_FIELDS = ("snapshot_heap_entries_read", "snapshot_retained_entries_read")


def _path_errors(result, state):
    previous = state["previous_settled_root_routed"]
    preparation, old_preparation = result["routing_preparation"], previous["routing_preparation"]
    phase, old_phase = result["root_settle_work"], previous["root_settle_work"]
    errors = []
    if (set(preparation) != set(ROUTING_COUNTERS)
            or any(type(value) is not int or value < 0 for value in preparation.values())):
        errors.append("fused root preparation must retain all11 actual nonnegative integer counters")
    raw_heap = old_preparation["root_snapshot_heap_entries_read"] // 2
    if (old_preparation["root_snapshot_heap_entries_read"] != 2 * raw_heap
            or preparation.get("root_snapshot_calls") != 1
            or preparation.get("root_snapshot_heap_entries_read") != raw_heap
            or preparation.get("root_snapshot_retained_entries_read") != 0
            or not _same({name: value for name, value in preparation.items() if name not in ROOT_SCAN_FIELDS},
                         {name: value for name, value in old_preparation.items() if name not in ROOT_SCAN_FIELDS})):
        errors.append("fused raw-root checkpoint must pay one raw heap pass and no retained-list pass")
    if old_phase is None:
        if phase is not None:
            errors.append("untriggered fused roots must retain a null settled checkpoint and null phase work")
    elif phase is None:
        errors.append("fused score-certified open roots must retain their actual settlement")
    else:
        if (set(phase) != set(ROOT_SETTLE_COUNTERS)
                or any(type(value) is not int or value < 0 for value in phase.values())):
            errors.append("fused settlement must retain all15 actual nonnegative integer counters")
        if (phase.get("snapshot_calls") != 1
                or phase.get("frontier_regions_before") != raw_heap
                or phase.get("snapshot_heap_entries_read") != phase.get("frontier_regions_after")
                or phase.get("snapshot_retained_entries_read") != 0
                or not _same({name: value for name, value in phase.items() if name not in SETTLED_SCAN_FIELDS},
                             {name: value for name, value in old_phase.items() if name not in SETTLED_SCAN_FIELDS})):
            errors.append("fused settlement must keep every native diagnostic and pay one actual post-settle raw heap pass")
    if not _same({name: value for name, value in result.items() if name not in ("routing_preparation", "root_settle_work")},
                 {name: value for name, value in previous.items() if name not in ("routing_preparation", "root_settle_work")}):
        errors.append("raw M, settled/final snapshots, actions, canonical proof, LP/probe path and all remaining fees must equal the genuine settled parent")
    traces = state["previous_settled_root_traces"]
    if (not _same(traces, state["previous_zero_traces"]) or not _same(traces, state["traces"]["new"])
            or set(traces) != {str(cap) for cap in state["caps"]}):
        errors.append("the genuine four-cap settled-source native ladder must remain complete and unchanged")
    return errors


def _audit_fused_root_result(result, state, protocol, model, lanes, label, expected):
    if label != "candidate":
        metric, errors, quality = _audit_settled_root_result(result, state, protocol, model, lanes,
            "candidate" if label == "settled" else label, expected)
        if label == "settled" and not _same(result, state["previous_settled_root_routed"]):
            errors.append("settled control differs from its actual frozen whole-selector result")
        return metric, errors, quality
    # Audit the genuine parent with its original paid scans. New fees stay on
    # the candidate and are checked independently, never substituted away.
    previous = state["previous_settled_root_routed"]
    _, errors, _ = _audit_settled_root_result(previous, state, protocol, model, lanes, "candidate", expected)
    errors.extend(_path_errors(result, state))
    budget = previous["final_lp"]["selection"]["lp_node_budget"]
    metric = _metric(result["selection"], state, model, lanes, budget, expected, result["final_lp"]["preprocessing"])
    if result["selection"]["lp_node_budget"] != budget or metric["failures"]:
        errors.append("fused checkpoint returned an invalid action, certificate or actual LP work")
    return metric, errors, _exact(result["selection"], metric, expected)


def _work(results, label):
    return _settled_work(results, "candidate" if label == "settled" else label)


def summarize(protocol, rows):
    parent_rows = [{**row, "states": [{**state, "routed": state["previous_settled_root_routed"]}
                                     for state in row["states"]]} for row in rows]
    result = _settled_summary(protocol, parent_rows)
    entries = [(row, state) for row in rows for state in row["states"]]
    path_failures = []
    for row, state in entries:
        errors = _path_errors(state["routed"], state)
        if errors:
            path_failures.append({"node": row["node"], "lanes": row["lanes"],
                **{name: state[name] for name in ("state_id", "queue", "timing_group")}, "errors": errors})
    result["failures"].extend(path_failures)
    states = [state for _, state in entries]
    executed = _work([state["routed"] for state in states], "candidate")
    result["executed"]["whole_selector"] = executed
    result["executed_routing_preparation_work"] = executed["routing_preparation"]
    result["executed_dual_face_work"] = executed["dual_face_work"]
    result["executed_root_settle_work"] = executed["root_settle_work"]
    result["root_settle_diagnostics"] = executed["root_settle_diagnostics"]
    borrowed = _work([state["previous_settled_root_routed"] for state in states], "settled")
    result["borrowed"]["whole_selector"]["settled"] = borrowed
    result["borrowed_settled_root_snapshots"] = sum(len(state["previous_settled_root_traces"]) for state in states)
    result["removed_snapshot_reads"] = {
        "raw_root": {name: borrowed["routing_preparation"][name] - executed["routing_preparation"][name] for name in ROOT_SCAN_FIELDS},
        "settled": {name: borrowed["root_settle_work"][name] - executed["root_settle_work"][name] for name in SETTLED_SCAN_FIELDS},
    }
    result["behavior_preserved"] = not path_failures
    for name, group in result["groups"].items():
        items = [state for state in states if state["timing_group"] == name]
        group["executed"]["whole_selector"] = _work([state["routed"] for state in items], "candidate")
        group["borrowed"]["whole_selector"]["settled"] = _work([state["previous_settled_root_routed"] for state in items], "settled")
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
            "FUSED_ROOT_DUAL_FACE_DEVELOPMENT_ADVANCE" if advance else "FUSED_ROOT_DUAL_FACE_DEVELOPMENT_HOLD")
    return result
