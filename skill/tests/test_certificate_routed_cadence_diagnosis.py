"""Cadence replay uses original ordering and charges actual implementations."""
from collections import Counter
from copy import deepcopy
import gc
from itertools import permutations
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_certificate_routed_cadence_diagnosis as diagnosis


def test_saved_sequences_compare_without_reencoding_or_ignoring_work():
    assert diagnosis.same_result({"counts": (1, 2), "work": {"n": 3}},
                                 {"counts": [1, 2], "work": {"n": 3}})
    assert not diagnosis.same_result({"counts": (1, 2)}, {"counts": [1, 3]})
    assert not diagnosis.same_result({"n": 3}, {"n": 3, "extra": 0})
    assert not diagnosis.same_result([1], [1, 2])


@pytest.fixture
def mocked(monkeypatch):
    from algorithm.experiments import resumable_probe_development_summary as candidate_work
    from algorithm.experiments import strict_score_proof_timing_summary as strict_work

    events, calls, now = [], Counter(), [0]
    expected = {}
    for label in diagnosis.LABELS:
        expected[label] = {"selection": {"lp_node_budget": 4096 if label == "long" else 144},
                           "paid_work": 1, "arm": label}
        if label == "long":
            expected[label]["preprocessing"] = {"arm": label}
        else:
            expected[label]["final_lp"] = {"preprocessing": {"arm": label}}
    states = [{"state_id": f"state{index}", "queue": [index, 1, 2], "target": index > 0,
               "reference": {"reference": index}, "oracle": None,
               "effective_reference_complete": True, "expected_results": deepcopy(expected)}
              for index in range(3)]
    label_orders = [list(permutations(diagnosis.LABELS)) for _ in states]
    state_orders = [[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1]]
    cell = {"node": "mock", "lanes": 64, "states": states,
            "orders": {"state_orders": state_orders, "label_orders": label_orders}}
    protocol = {"registered_at_utc": "frozen", "repeats": 6, "warmups_per_label": 2,
                "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
                "long_lp_budget": 4096, "control_integer_state_budget": 100000}
    outputs = deepcopy(expected)

    def selector(label):
        def call(options, queue, lanes, *budgets):
            events.append(("selector", label, queue[0]))
            calls[label] += 1
            now[0] += 10
            assert lanes == 64
            assert budgets == ((4096, 100000) if label == "long" else (144, 256, 8192))
            return deepcopy(outputs[label])
        return call

    def serialize(value):
        events.append(("serialize", value["arm"]))
        return deepcopy(value)

    def record(metrics, selected, model, queue, lanes, budget, expected_key, prep, reference):
        events.append(("record", prep["arm"], queue[0]))
        assert expected_key == ("reference", queue[0])
        assert reference == {"reference": queue[0]}
        assert prep["arm"] == ("candidate" if queue[0] > 0 else prep["arm"])
        metrics["calls"] += 1
        now[0] += 1

    monkeypatch.setattr(diagnosis, "asdict", serialize)
    monkeypatch.setattr(diagnosis, "selection_from_record", lambda value: value)
    monkeypatch.setattr(diagnosis, "runtime_provenance", lambda: {"cpu_affinity": [0], "gc_enabled": gc.isenabled()})
    monkeypatch.setattr(diagnosis, "perf_counter_ns", lambda: now[0])
    monkeypatch.setattr(diagnosis, "process_time_ns", lambda: now[0])
    monkeypatch.setattr(gc, "collect", lambda: pytest.fail("manual GC changes the replay workload"))
    summarize = lambda values: {"calls": len(values), "paid_work": sum(value["paid_work"] for value in values)}
    monkeypatch.setattr(candidate_work, "_candidate_work", summarize)
    monkeypatch.setattr(strict_work, "_work", summarize)
    runtime = SimpleNamespace(_metrics=lambda: {"calls": 0}, _record=record,
                              reference_key=lambda reference, *args: ("reference", reference["reference"]))
    return SimpleNamespace(cell=cell, protocol=protocol, runtime=runtime,
        model=SimpleNamespace(node="mock", options=(1, 2)), outputs=outputs, events=events,
        calls=calls, selectors={label: selector(label) for label in diagnosis.LABELS})


def run(mocked):
    return diagnosis.run_cell(mocked.model, mocked.cell, mocked.protocol, mocked.runtime,
                              selectors=mocked.selectors, progress=False)


def test_warmups_are_immediate_then_timed_blocks_remain_adjacent(mocked):
    row = run(mocked)
    cursor = 0
    for _ in range(2):
        for role in diagnosis.LABELS:
            assert mocked.events[cursor:cursor + 3] == [
                ("selector", role, 0), ("serialize", role), ("record", role, 0)]
            cursor += 3
    for repeat, state_order in enumerate(mocked.cell["orders"]["state_orders"]):
        for index in state_order:
            roles = mocked.cell["orders"]["label_orders"][index][repeat]
            implementations = ["candidate" if index else role for role in roles]
            assert mocked.events[cursor:cursor + 3] == [("selector", arm, index) for arm in implementations]
            cursor += 3
            for arm in implementations:
                assert mocked.events[cursor:cursor + 2] == [("serialize", arm), ("record", arm, index)]
                cursor += 2
    assert cursor == len(mocked.events)
    assert all(call["wall_ns"] == call["cpu_ns"] == 10 for state in row["states"] for call in state["calls"])


def test_target_roles_share_candidate_but_background_keeps_all_controls(mocked):
    row = run(mocked)
    assert mocked.calls == {"candidate": 44, "probe": 8, "long": 8}
    assert {label: value["calls"] for label, value in row["work"].items()} == mocked.calls
    assert row["work"]["candidate"]["paid_work"] == 44
    for index, state in enumerate(row["states"]):
        assert len(state["calls"]) == (24 if index == 0 else 18)
        assert all(value["calls"] == (8 if index == 0 else 6) for value in state["metrics"].values())
        assert all(call["matches_frozen"] for call in state["calls"])
        assert not state["mismatches"]
        if index:
            assert {call["implementation"] for call in state["calls"]} == {"candidate"}
            assert {result["arm"] for result in state["first_results"].values()} == {"candidate"}


def test_relative_block_starts_recover_registered_state_order(mocked):
    row = run(mocked)
    blocks = sorted((block["start_ns"], index, block) for index, state in enumerate(row["states"])
                    for block in state["blocks"])
    assert [entry[2]["kind"] for entry in blocks[:6]] == ["warmup"] * 6
    assert [(block["repeat"], index) for _, index, block in blocks[6:]] == [
        (repeat, index) for repeat, order in enumerate(mocked.cell["orders"]["state_orders"]) for index in order]
    assert [start for start, _, _ in blocks] == sorted({start for start, _, _ in blocks})
    for state in row["states"][1:]:
        assert all(len(block["order"]) == 3 for block in state["blocks"])
        starts = [block["start_ns"] for block in state["blocks"]]
        assert all(right > left for left, right in zip(starts, starts[1:]))


@pytest.mark.parametrize("role", diagnosis.LABELS)
def test_mismatches_are_not_dropped_or_replaced_and_actual_work_is_charged(mocked, role):
    mocked.outputs[role]["paid_work"] = 7
    row = run(mocked)
    all_calls = [call for state in row["states"] for call in state["calls"]]
    mismatches = [result for state in row["states"] for result in state["mismatches"]]
    expected_count = 44 if role == "candidate" else 8
    assert len(all_calls) == 60
    assert len(mismatches) == sum(not call["matches_frozen"] for call in all_calls) == expected_count
    assert all(value["result"]["paid_work"] == 7 and value["implementation"] == role for value in mismatches)
    assert row["work"][role] == {"calls": expected_count, "paid_work": 7 * expected_count}


def test_reference_and_oracle_keys_are_reused_before_selectors(mocked, monkeypatch):
    events = []
    mocked.cell["states"][1]["oracle"] = {"value": "frozen oracle"}
    mocked.runtime.reference_key = lambda ref, *args: events.append(("reference", ref["reference"])) or (
        None if ref["reference"] == 1 else ("reference", ref["reference"]))
    monkeypatch.setattr(diagnosis, "oracle_key", lambda value, count: events.append(("oracle", value)) or ("reference", 1))
    original = mocked.selectors["candidate"]
    mocked.selectors["candidate"] = lambda *args: events.append(("selector",)) or original(*args)
    run(mocked)
    assert events[:4] == [("reference", 0), ("reference", 1), ("oracle", {"value": "frozen oracle"}), ("reference", 2)]
    assert events[4] == ("selector",)


def test_verification_rejects_changed_registered_order_without_execution(monkeypatch):
    model = object()
    frozen = {"cells": [{"orders": [[0, 1]]}], "repeats": 6}
    monkeypatch.setattr(diagnosis, "build_protocol", lambda models: deepcopy(frozen))
    protocol = {**deepcopy(frozen), "registered_at_utc": "registered"}
    diagnosis.verify_protocol(protocol, [model])
    protocol["cells"][0]["orders"][0].reverse()
    with pytest.raises(ValueError, match="changed"):
        diagnosis.verify_protocol(protocol, [model])
