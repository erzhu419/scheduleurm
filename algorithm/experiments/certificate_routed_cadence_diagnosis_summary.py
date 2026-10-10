"""Two-state A/A readout under the frozen full timing workload; retain prior HOLD."""
from __future__ import annotations

from collections import Counter
from statistics import median


RUNTIME_KEYS = ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")


def _identity(state):
    return state["state_id"], tuple(state["queue"]), state["target"]


def _combine_work(items):
    """Add charged work once; maxima remain maxima across cells."""
    result = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict):
                result[key] = _combine_work((result.get(key, {}), value))
            elif key.startswith("max_"):
                result[key] = max(result.get(key, 0), value)
            else:
                result[key] = result.get(key, 0) + value
    return result


def _expected_blocks(protocol, cell, index):
    warmup = [("warmup", repeat, [role]) for repeat in range(protocol["warmups_per_label"])
              for role in protocol["labels"]] if index == 0 else []
    return warmup + [("timed", repeat, cell["orders"]["label_orders"][index][repeat])
                     for repeat in range(protocol["repeats"])]


def _validate(protocol, rows):
    errors = []
    if [(row["node"], row["lanes"]) for row in rows] != [
            (cell["node"], cell["lanes"]) for cell in protocol["cells"]]:
        return ["cell roster or execution order differs from registration"]
    if not protocol["prior_quality_passed"]:
        errors.append("frozen reference quality did not pass")
    calls = [call for row in rows for state in row["states"] for call in state["calls"]]
    kinds, implementations = Counter(call["kind"] for call in calls), Counter(call["implementation"] for call in calls)
    if (len(calls) != protocol["expected_selector_calls"]
            or kinds != {"timed": protocol["expected_timed_calls"], "warmup": protocol["expected_warmup_calls"]}
            or implementations != protocol["expected_implementation_calls"]):
        errors.append("selector counts differ from the registered workload")
    for cell, row in zip(protocol["cells"], rows, strict=True):
        prefix = f'{cell["node"]}:R{cell["lanes"]}: '
        provenance = [row[key] for key in RUNTIME_KEYS]
        if any(item["cpu_affinity"] != [protocol["cpu"]] or not item["gc_enabled"]
               or any(value != "1" for value in item["thread_environment"].values()) for item in provenance):
            errors.append(prefix + "affinity, GC or thread environment differs from registration")
        stable = ("hostname", "platform", "python", "scipy", "cpu_model", "gc_threshold")
        if any(any(item[key] != provenance[0][key] for key in stable) for item in provenance):
            errors.append(prefix + "runtime provenance changed during the cell")
        if list(map(_identity, row["states"])) != list(map(_identity, cell["states"])):
            errors.append(prefix + "state roster differs from registration")
            continue
        chronology, implementation_counts = [], Counter()
        for index, (state, actual) in enumerate(zip(cell["states"], row["states"], strict=True)):
            name = prefix + state["state_id"] + ": "
            expected_blocks = _expected_blocks(protocol, cell, index)
            blocks = [(block["kind"], block["repeat"], block["order"]) for block in actual["blocks"]]
            if blocks != expected_blocks:
                errors.append(name + "missing, duplicate or reordered registered blocks")
            expected_calls = [(kind, repeat, role) for kind, repeat, order in expected_blocks for role in order]
            calls = [(call["kind"], call["repeat"], call["role"]) for call in actual["calls"]]
            if calls != expected_calls:
                errors.append(name + "missing, duplicate or reordered registered calls")
            if not state["effective_reference_complete"]:
                errors.append(name + "independent frozen reference is incomplete")
            if actual["mismatches"] or any(not call["matches_frozen"] for call in actual["calls"]):
                errors.append(name + "selector output differs from its frozen result")
            for role in protocol["labels"]:
                implementation = "candidate" if state["target"] else role
                result = state["expected_results"][implementation]
                metric = actual["metrics"][role]
                count = sum(call["role"] == role for call in actual["calls"])
                if (actual["first_results"].get(role) != result
                        or not all(result["selection"][key] for key in ("exact", "score_exact", "canonical_exact"))
                        or any(metric[key] != count for key in ("calls", "score_exact_count", "canonical_exact_count", "reference_comparable_count"))
                        or any(metric[key] for key in ("failures", "score_mismatch_count", "canonical_action_mismatch_count"))):
                    errors.append(name + role + " result or quality is invalid")
            for call in actual["calls"]:
                implementation_counts[call["implementation"]] += 1
                if call["implementation"] != ("candidate" if state["target"] else call["role"]):
                    errors.append(name + "role used the wrong implementation")
                if any(type(call[key]) is not int or call[key] < minimum
                       for key, minimum in (("wall_ns", 1), ("cpu_ns", 0))):
                    errors.append(name + "invalid selector duration")
            for block in actual["blocks"]:
                chronology.append((block["start_ns"], block["kind"], block["repeat"], index, block["order"]))
        actual_order = [(kind, repeat, index, order) for _, kind, repeat, index, order in sorted(chronology)]
        expected_order = [(kind, repeat, 0, order) for kind, repeat, order in _expected_blocks(protocol, cell, 0)
                          if kind == "warmup"]
        expected_order += [("timed", repeat, index, cell["orders"]["label_orders"][index][repeat])
                          for repeat, indices in enumerate(cell["orders"]["state_orders"]) for index in indices]
        if actual_order != expected_order:
            errors.append(prefix + "global block chronology differs from registered workload cadence")
        if any(row["work"][role]["calls"] != implementation_counts[role] for role in protocol["labels"]):
            errors.append(prefix + "charged work calls differ from actual implementations")
    return errors


def _target_readout(cell, state):
    calls = [call for call in state["calls"] if call["kind"] == "timed"]
    roles = {role: {clock: [call[clock] for call in calls if call["role"] == role]
                   for clock in ("wall_ns", "cpu_ns")} for role in ("candidate", "probe", "long")}
    arms = {role: {"wall_ms": median(values["wall_ns"]) / 1e6,
                   "cpu_ms": median(values["cpu_ns"]) / 1e6, **values} for role, values in roles.items()}
    ratios = {clock: {role: median(roles["candidate"][clock]) / median(roles[role][clock])
                     if median(roles[role][clock]) else None for role in ("probe", "long")}
              for clock in ("wall_ns", "cpu_ns")}
    wall = ratios["wall_ns"]
    times = [block["start_ns"] for block in state["blocks"] if block["kind"] == "timed"]
    medians = [median(values["wall_ns"]) for values in roles.values()]
    return {"node": cell["node"], "lanes": cell["lanes"], "state_id": state["state_id"], "queue": state["queue"],
        "arms": arms, "candidate_over_control_wall_ratio": wall,
        "candidate_over_control_cpu_ratio": ratios["cpu_ns"],
        "apparent_slowdown_over_5pct": {role: ratio > 1.05 for role, ratio in wall.items()},
        "any_apparent_slowdown_over_5pct": any(ratio > 1.05 for ratio in wall.values()),
        "symmetric_wall_max_min_ratio": max(medians) / min(medians),
        "block_start_ns": times, "consecutive_block_gaps_s": [(later - earlier) / 1e9 for earlier, later in zip(times, times[1:])]}


def summarize(protocol, rows):
    errors = _validate(protocol, rows)
    calls = [call for row in rows for state in row["states"] for call in state["calls"]]
    result = {"decision": "INVALID_EXECUTION" if errors else "CADENCE_DIAGNOSIS_COMPLETE",
        "execution_checks_passed": not errors, "execution_failures": errors,
        "prior_decision": protocol["prior_decision"], "prior_hold_retained": True,
        "executed_selector_calls": len(calls), "call_counts": dict(Counter(call["kind"] for call in calls)),
        "selector_calls_by_implementation": dict(Counter(call["implementation"] for call in calls)),
        "executed_reference_calls": 0, "executed_oracle_calls": 0,
        "borrowed_reference_count": sum(len(cell["states"]) for cell in protocol["cells"]),
        "work": {role: _combine_work(row["work"][role] for row in rows) for role in protocol["labels"]},
        "elapsed_s": sum(row["elapsed_s"] for row in rows), "targets": [],
        "runtime": [{"node": row["node"], "lanes": row["lanes"], **{key: row[key] for key in RUNTIME_KEYS}} for row in rows],
        "limitations": protocol["limitations"]}
    if errors:
        return result
    targets = [_target_readout(row, state) for row in rows for state in row["states"] if state["target"]]
    result["targets"] = targets
    result["aa_readout"] = {
        "target_states": len(targets),
        "states_with_apparent_slowdown": sum(state["any_apparent_slowdown_over_5pct"] for state in targets),
        "states_with_symmetric_spread_over_5pct": sum(state["symmetric_wall_max_min_ratio"] > 1.05 for state in targets),
        "max_candidate_over_control_wall_ratio": max(ratio for state in targets for ratio in state["candidate_over_control_wall_ratio"].values()),
        "max_symmetric_wall_max_min_ratio": max(state["symmetric_wall_max_min_ratio"] for state in targets),
        "interpretation": "Fixed candidate role versus probe/long uses the prior >5% direction. Symmetric spread is separate. Two selected states in one replay do not estimate an independent error rate or universal timing resolution. Background calls establish workload accounting, not performance improvement. Prior HOLD is retained."}
    return result
