"""Frozen four-arm order, borrowed references, and whole-call timing blocks."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
from types import SimpleNamespace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import CanonicalCountSpaceBnbSelection
from algorithm.experiments import aligned_hardware_branch_reserved_timing as runner


LABELS = ("candidate", "routed", "probe", "long")


def test_williams_orders_balance_positions_adjacencies_and_every_state():
    actual = runner.timing_orders("toy", 8, 5)
    assert actual == runner.timing_orders("toy", 8, 5)
    assert len(actual["label_orders"]) == 5 and len(actual["state_orders"]) == 8
    expected = Counter({("candidate", "routed", "long", "probe"): 2,
        ("routed", "probe", "candidate", "long"): 2,
        ("probe", "long", "routed", "candidate"): 2,
        ("long", "candidate", "probe", "routed"): 2})
    for orders in actual["label_orders"]:
        assert Counter(map(tuple, orders)) == expected
        for position in range(4):
            assert Counter(order[position] for order in orders) == Counter(dict.fromkeys(LABELS, 2))
        assert Counter((left, right) for order in orders for left, right in zip(order, order[1:])) == \
            Counter({(left, right): 2 for left in LABELS for right in LABELS if left != right})
    assert all(sorted(order) == list(range(5)) for order in actual["state_orders"])


@dataclass(frozen=True)
class Stage:
    selection: CanonicalCountSpaceBnbSelection
    preprocessing: dict
    capacity_propagation: dict


@dataclass(frozen=True)
class Result:
    selection: CanonicalCountSpaceBnbSelection
    initial_lp: Stage
    final_lp: Stage
    preprocessing: dict
    label: str


def selection(queue, cap):
    values = {field.name: 0 for field in fields(CanonicalCountSpaceBnbSelection)}
    active = 0 if queue == (1, 0) else 1
    duration = 1.0 if active == 0 else 2.0
    values.update(option_counts=(1, 0) if active == 0 else (0, 1), option_indices=(active, None),
        service=queue, duration=duration, score=1.0 / duration, upper_bound=1.0 / duration,
        exact=True, score_exact=True, canonical_exact=True, lp_node_budget=cap, lp_calls=1,
        duration_root_count=2)
    return CanonicalCountSpaceBnbSelection(**values)


def payload(label, queue):
    selected = selection(queue, 4096 if label == "long" else 144)
    stage = Stage(selected, {"original_option_count": 2}, {"region_visits": 2})
    return Result(selected, stage, stage, stage.preprocessing, label)


def call_harness(monkeypatch, *, vary=False):
    events, calls, counts = [], [], Counter()
    queues = ((1, 0), (0, 1))
    saved = []
    for index, queue in enumerate(queues):
        saved.append({"state_id": f"state{index}", "queue": list(queue),
            "reference": asdict(selection(queue, 4096)), "reference_metrics": {"borrowed": index},
            "reference_complete": True, "oracle": None, "effective_reference_complete": True,
            "effective_reference_source": "full_K4096",
            "routed": asdict(payload("candidate", queue)), "old_routed": asdict(payload("routed", queue)),
            "traces": {"new": {"144": asdict(payload("candidate", queue).initial_lp)},
                       "old": {"144": asdict(payload("routed", queue).initial_lp)}}})
    protocol = {"registered_at_utc": "timing", "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "long_lp_budget": 4096, "control_integer_state_budget": 100000,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "reference_source": {"path": "source.json", "registered_at_utc": "source"}}
    states = [{"state_id": state["state_id"], "queue": state["queue"], "source_index": index,
               "source_state_id": state["state_id"], "cohort": "fresh_confirmation" if index == 0 else "historical_trigger"}
              for index, state in enumerate(saved)]
    cell = {"node": "toy", "lanes": 2, "states": states, "orders": runner.timing_orders("toy", 2, 2)}
    model = SimpleNamespace(node="toy", lanes=1, options=(SimpleNamespace(requirements=(1, 0), duration=1.0),
                                                        SimpleNamespace(requirements=(0, 1), duration=2.0)))

    def selector(label):
        def run(*args):
            events.append("selector")
            calls.append((label, args))
            queue = tuple(args[1])
            counts[label, queue] += 1
            result = payload(label, queue)
            if vary and label == "candidate" and queue == queues[0] and counts[label, queue] in (1, 4):
                incomplete = replace(result.selection, exact=False, canonical_exact=False, frontier_regions_remaining=1)
                stage = replace(result.initial_lp, selection=incomplete)
                result = replace(result, selection=incomplete, initial_lp=stage, final_lp=stage)
            return result
        return run

    def record(metric, selected, *args):
        events.append("audit")
        metric["calls"] += 1
        metric["unclosed"] += int(not selected.canonical_exact)

    def unexpected(*args, **kwargs):
        pytest.fail("timing reuses references and must not solve new references or oracles")

    def load(path, node, lanes, registration):
        assert (path, node, lanes, registration) == (runner.ROOT / "source.json", "toy", 2, "source")
        return {"states": saved}

    clocks = {"wall": 0, "cpu": 0}

    def clock(label):
        events.append(label)
        clocks[label] += 100
        return clocks[label]

    original_asdict = runner.asdict

    def serialize(result):
        events.append("serialize")
        return original_asdict(result)

    monkeypatch.setattr(runner, "load_cell", load)
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": True})
    monkeypatch.setattr(runner, "asdict", serialize)
    monkeypatch.setattr(runner, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(runner, "process_time_ns", lambda: clock("cpu"))
    from algorithm.experiments import aligned_hardware_strict_score_proof_development as references
    from algorithm.experiments import integer_capacity_canonical_oracle as oracles
    monkeypatch.setattr(references, "_new_reference", unexpected)
    monkeypatch.setattr(oracles, "canonical_at_certified_score", unexpected)
    runtime = SimpleNamespace(_metrics=lambda: {"calls": 0, "unclosed": 0}, _record=record,
        reference_key=lambda reference, queue, count: ("canonical",) if reference.canonical_exact else None,
        select_canonical_count_space_bnb_frame=unexpected)
    selectors = {label: selector(label) for label in LABELS}
    return model, cell, protocol, runtime, selectors, events, calls, saved


def test_adjacent_whole_calls_precede_serialization_and_warmups_are_not_timed_metrics(monkeypatch):
    model, cell, protocol, runtime, selectors, events, calls, saved = call_harness(monkeypatch)
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert len(calls) == 8 + 2 * 8 * 4
    assert [label for label, _ in calls[:8]] == list(LABELS) * 2
    expected_order = [(label, tuple(cell["states"][index]["queue"]))
        for repeat, order in enumerate(cell["orders"]["state_orders"]) for index in order
        for label in cell["orders"]["label_orders"][index][repeat]]
    assert [(label, tuple(args[1])) for label, args in calls[8:]] == expected_order
    for label, args in calls:
        assert args[0] is model.options and args[2] == 2
        assert args[3:] == ((4096, 100000) if label == "long" else (144, 256, 8192))
    timed_start = [index for index, event in enumerate(events) if event == "selector"][8]
    measured, completed = events[timed_start - 2:], 0
    for index, event in enumerate(measured):
        if event == "selector":
            assert measured[index - 2:index + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
            completed += 1
        elif event in ("serialize", "audit"):
            assert completed % 4 == 0
    assert completed == 64
    assert len(row["warmups"]) == 8 and all(warm["metrics"]["calls"] == 1 for warm in row["warmups"])
    assert row["state_orders"] == cell["orders"]["state_orders"]
    assert (row["new_reference_calls"], row["new_oracle_calls"], row["screen_selector_calls"]) == (0, 0, 0)
    assert row["borrowed_reference_count"] == 2
    for index, state in enumerate(row["states"]):
        assert all(state["metrics"][label]["calls"] == 8 for label in LABELS)
        assert all(state["wall_ns"][label] == state["cpu_ns"][label] == [100] * 8 for label in LABELS)
        assert state["blocks"] == [{"repeat": repeat, "order": cell["orders"]["label_orders"][index][repeat]}
                                   for repeat in range(8)]
        assert state["expected_results"] == {"candidate": json.loads(json.dumps(saved[index]["routed"])),
                                             "routed": json.loads(json.dumps(saved[index]["old_routed"]))}
        for key in ("reference", "reference_metrics", "reference_complete", "oracle",
                    "effective_reference_complete", "effective_reference_source"):
            assert state[key] == json.loads(json.dumps(saved[index][key]))


def test_warmup_nonclosure_and_one_repeat_mismatch_remain_separate(monkeypatch):
    model, cell, protocol, runtime, selectors, _, calls, saved = call_harness(monkeypatch, vary=True)
    saved[0]["effective_reference_complete"] = False
    saved[0]["effective_reference_source"] = None
    saved[0]["reference"]["canonical_exact"] = False
    saved[0]["reference"]["exact"] = False
    saved[0]["reference_complete"] = False
    row = runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert len(calls) == 72
    assert row["warmups"][0]["result"]["selection"]["canonical_exact"] is False
    assert row["warmups"][0]["metrics"]["unclosed"] == 1
    state = row["states"][0]
    assert state["effective_reference_complete"] is False
    assert state["first_results"]["candidate"]["selection"]["canonical_exact"] is True
    mismatch = state["repeat_mismatches"]["candidate"]
    assert len(mismatch) == 1 and mismatch[0]["repeat_index"] == 1
    assert mismatch[0]["result"]["selection"]["canonical_exact"] is False
    assert state["metrics"]["candidate"] == {"calls": 8, "unclosed": 1}
    assert all(len(other["wall_ns"][label]) == 8 for other in row["states"] for label in LABELS)


@pytest.mark.parametrize("changed", ("source_index", "queue"))
def test_wrong_borrowed_state_stops_before_any_selector(monkeypatch, changed):
    model, cell, protocol, runtime, selectors, _, calls, saved = call_harness(monkeypatch)
    if changed == "source_index":
        cell["states"][0]["source_index"] = 1
    else:
        saved[0]["queue"] = [9, 9]
    with pytest.raises(ValueError, match="borrowed timing state"):
        runner.run_cell(model, cell, protocol, runtime, selectors, progress=False)
    assert not calls


def test_locked_roster_uses_first_sixteen_per_panel_and_all_original_subgroup_keys(monkeypatch, tmp_path):
    from skill.tests.test_branch_reserved_confirmation import protocol_fixture
    from algorithm.experiments import aligned_hardware_branch_reserved_confirmation as confirmation
    from algorithm.experiments import aligned_hardware_shared_threshold_queue_confirmation as sampler
    models, old_root_protocol, _ = protocol_fixture(monkeypatch, tmp_path)
    source = confirmation.build_protocol(models)
    source["registered_at_utc"] = "work confirmation"
    triggers = old_root_protocol["historical_trigger_keys"]
    limits = {"fresh_nonregression_limit": 1.05, "fresh_speedup_limit": .95,
        "score_gain_limit": .95, "state_ratio_limit": 1.05, "nonregression_limit": 1.05,
        "trigger_long_ratio_limit": 1.0, "tail_state_limit": 1.05}
    previous = {**{key: source[key] for key in ("models", "execution_sources", "initial_lp_budget", "lp_node_budget",
        "integer_state_budget", "reference_budget", "oracle_state_budget")}, **limits,
        "long_lp_budget": 4096, "control_integer_state_budget": 100000,
        "historical_trigger_keys": triggers, "score_unproved_keys": triggers[:6],
        "previous_untimed_probe_keys": triggers[6:10]}
    current_path, previous_path = tmp_path / "current.json", tmp_path / "old_timing.json"
    current_path.write_text(json.dumps({"decision": "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE",
        "confirmation_gate_passed": True, "prior_decisions": source["prior_decisions"]}))
    current_path.with_name("current_protocol.json").write_text(json.dumps(source))
    previous_path.write_text(json.dumps({"decision": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD", **{
        key: True for key in ("certificate_checks_passed", "effective_references_complete",
            "candidate_quality_gate_passed", "comparison_quality_gate_passed", "execution_checks_passed")}}))
    previous_path.with_name("old_timing_protocol.json").write_text(json.dumps(previous))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_OUTPUT", current_path)
    monkeypatch.setattr(runner, "PREVIOUS_OUTPUT", previous_path)
    monkeypatch.setattr(runner, "SOURCES", ())

    def unexpected(*args, **kwargs):
        pytest.fail("timing must use the locked confirmation queues without new sampling")

    monkeypatch.setattr(sampler, "sample_queues", unexpected)
    protocol = runner.build_protocol(models)
    states = [state for cell in protocol["cells"] for state in cell["states"]]
    assert len(states) == 1058
    assert sum(state["source"] == "fresh_confirmation" for state in states) == 1024
    assert sum(state["source"] == "historical_regression" for state in states) == 34
    assert sum(state["score_unproved_regression"] for state in states) == 6
    assert sum(state["previous_integer_trigger"] and not state["score_unproved_regression"] for state in states) == 12
    assert sum(state["cohort"] == "ordinary_tail_regression" for state in states) == 16
    current_cells = {(cell["node"], cell["lanes"]): cell for cell in source["cells"]}
    for cell in protocol["cells"]:
        old = current_cells[cell["node"], cell["lanes"]]["states"]
        first_ids = {state["state_id"] for panel in source["queue_panels"][str(cell["lanes"])]
                     for state in panel["states"][:16]}
        expected = [index for index, state in enumerate(old) if state["source"] == "historical_regression"
                    or state["source"] == "fresh_confirmation" and state["state_id"] in first_ids]
        assert [state["source_index"] for state in cell["states"]] == expected
        assert all((state["source_state_id"], state["queue"]) ==
                   (old[state["source_index"]]["state_id"], old[state["source_index"]]["queue"])
                   for state in cell["states"])
        assert all(state["source"] != "new_zero" for state in cell["states"])
    assert protocol["expected_timed_calls"] == 33856 and protocol["expected_warmup_calls"] == 128
    assert protocol["reference_source"] == {"path": "current.json", "registered_at_utc": "work confirmation"}
    assert protocol["prior_decisions"]["branch_reserved_confirmation"] == "BRANCH_RESERVED_FRESH_CONFIRMATION_ADVANCE"
    registered_protocol = {**protocol, "registered_at_utc": "timing registration"}
    runner.verify_protocol(registered_protocol, models)
    changed = deepcopy(registered_protocol)
    changed["cells"][0]["orders"]["label_orders"][0][0].reverse()
    with pytest.raises(ValueError):
        runner.verify_protocol(changed, models)
