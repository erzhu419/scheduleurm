"""A/A identity, adjacent timing boundaries and actual-work retention."""
from collections import Counter
from copy import deepcopy
import gc
from itertools import permutations
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_certificate_routed_aa_diagnosis as diagnosis


def test_each_state_window_visits_all_six_orders_once():
    orders = diagnosis.aa_orders()
    assert orders == diagnosis.aa_orders()
    assert len(orders["state_orders"]) == 12
    for window in range(12):
        assert len(orders["state_orders"][window]) == 6
        assert all(sorted(order) == [0, 1] for order in orders["state_orders"][window])
        for state in range(2):
            assert Counter(map(tuple, orders["label_orders"][state][window])) == {
                order: 1 for order in permutations(("a", "b", "c"))}


def test_phase_orders_are_three_ab_and_three_ba_per_state():
    orders = diagnosis.balanced_orders(2, diagnosis.PHASE_LABELS, 6, 77)
    for values in orders["label_orders"]:
        assert Counter(map(tuple, values)) == {("candidate", "probe"): 3, ("probe", "candidate"): 3}
    with pytest.raises(ValueError, match="permutation"):
        diagnosis.balanced_orders(2, diagnosis.LABELS, 5, 77)


def test_aa_labels_share_one_callable_identity():
    selector = lambda: None
    labels = diagnosis.aa_selectors(selector)
    assert set(labels) == {"a", "b", "c"}
    assert all(call is selector for call in labels.values())


def test_uninstrumented_block_times_every_complete_call_without_gc(monkeypatch):
    events = []
    now = [0]
    monkeypatch.setattr(diagnosis, "perf_counter_ns", lambda: now[0])
    monkeypatch.setattr(diagnosis, "process_time_ns", lambda: now[0])
    monkeypatch.setattr(gc, "collect", lambda: pytest.fail("manual GC changes the A/A protocol"))

    def call():
        events.append("called")
        now[0] += 10
        return object()

    enabled = gc.isenabled()
    outputs, metrics = diagnosis.run_block(["c", "a", "b"], diagnosis.aa_selectors(call))
    assert list(outputs) == ["c", "a", "b"]
    assert len({id(value) for value in outputs.values()}) == 3
    assert events == ["called"] * 3
    assert all(value == {"wall_ns": 10, "cpu_ns": 10} for value in metrics.values())
    assert gc.isenabled() == enabled


@pytest.mark.parametrize("raises", [False, True])
def test_routed_probe_patch_records_primary_phase_and_restores(monkeypatch, raises):
    from algorithm import certificate_routed_frame_candidates as routed
    from algorithm import resumable_probe_frame_candidates as probe
    from algorithm import resumable_integer_capacity_frame_candidates as engine

    monkeypatch.setattr(routed, "_probe", lambda: "routed probe")
    previous = routed._probe, probe._probe, engine.IntegerCapacitySearch.advance_to
    callbacks = list(gc.callbacks)

    def run():
        with diagnosis.RoutedInstrumentation() as instrument:
            result, measured = instrument.capture("candidate", routed._probe)
            assert result == "routed probe"
            assert measured["phases"]["probe"]["calls"] == 1
            for clock in ("wall_ns", "cpu_ns"):
                assert measured["primary_phase_" + clock] == measured["phases"]["probe"][clock]
            assert len(gc.callbacks) == len(callbacks) + 1
            if raises:
                raise RuntimeError("failure inside phase diagnosis")

    if raises:
        with pytest.raises(RuntimeError, match="phase diagnosis"):
            run()
    else:
        run()
    assert (routed._probe, probe._probe, engine.IntegerCapacitySearch.advance_to) == previous
    assert gc.callbacks == callbacks


@pytest.fixture
def synthetic(monkeypatch):
    from algorithm import certificate_routed_frame_candidates as routed
    from algorithm import resumable_probe_frame_candidates as probe
    from algorithm.experiments import resumable_probe_development_summary as development

    events = []
    selection = {"exact": True, "score_exact": True, "canonical_exact": True}
    expected = {label: {"selection": dict(selection), "arm": label, "paid_work": 1}
                for label in diagnosis.PHASE_LABELS}
    outputs = deepcopy(expected)
    metrics = {"failures": 0, "score_mismatch_count": 0, "canonical_action_mismatch_count": 0}
    states = [{"name": name, "node": "mock", "lanes": 64, "queue": queue,
               "expected_results": deepcopy(expected), "effective_reference_complete": True,
               "screen_metrics": {label: dict(metrics) for label in diagnosis.PHASE_LABELS}}
              for name, _, _, queue in diagnosis.CASES]
    protocol = {"states": states, "labels": diagnosis.LABELS, "phase_labels": diagnosis.PHASE_LABELS,
        "windows": 12, "repeats": 6, "warmups": 2, "phase_repeats": 6, "phase_warmups": 2,
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "aa_orders": diagnosis.aa_orders(),
        "aa_warmup_orders": diagnosis.warmup_orders(2, diagnosis.LABELS, 2, 33),
        "phase_orders": diagnosis.balanced_orders(2, diagnosis.PHASE_LABELS, 6, 34),
        "phase_warmup_orders": diagnosis.warmup_orders(2, diagnosis.PHASE_LABELS, 2, 35)}
    callbacks = len(gc.callbacks)

    def selector(label):
        def call(*args):
            events.append(("selector", label, len(gc.callbacks)))
            assert args[2:] == (64, 144, 256, 8192)
            return deepcopy(outputs[label])
        return call

    def serialize(result):
        events.append(("serialize", result["arm"], len(gc.callbacks)))
        return deepcopy(result)

    validate = diagnosis.validate_result

    def audit(*args):
        events.append(("validate", args[0]["arm"], len(gc.callbacks)))
        return validate(*args)

    monkeypatch.setattr(routed, "select_certificate_routed_frame", selector("candidate"))
    monkeypatch.setattr(probe, "select_resumable_probe_frame", selector("probe"))
    monkeypatch.setattr(diagnosis, "asdict", serialize)
    monkeypatch.setattr(diagnosis, "validate_result", audit)
    monkeypatch.setattr(diagnosis, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    monkeypatch.setattr(gc, "collect", lambda: pytest.fail("manual GC is forbidden in both stages"))
    monkeypatch.setattr(development, "_candidate_work",
                        lambda values: {"calls": len(values), "paid_work": sum(v["paid_work"] for v in values)})
    return SimpleNamespace(protocol=protocol, states=states, outputs=outputs, events=events,
                           callbacks=callbacks, models=[SimpleNamespace(node="mock", options=())])


def test_complete_blocks_precede_serialization_and_phase_is_separate(synthetic):
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    # 148 A/A blocks of three, then 16 phase blocks of two. No observers in A/A.
    cursor = 0
    for size, block_count, callbacks in ((3, 148, synthetic.callbacks), (2, 16, synthetic.callbacks + 1)):
        for _ in range(block_count):
            block = synthetic.events[cursor:cursor + size * 3]
            assert [event[0] for event in block] == ["selector"] * size + ["serialize", "validate"] * size
            assert all(event[2] == callbacks for event in block)
            cursor += size * 3
    assert cursor == len(synthetic.events)
    assert record["work"] == {"candidate": {"calls": 460, "paid_work": 460},
                              "probe": {"calls": 16, "paid_work": 16}}
    assert sum(len(row["calls"]) for row in record["states"]) == 476
    assert sum(len(row["blocks"]) for row in record["states"]) == 164
    for row in record["states"]:
        assert set(row["first_results"]) == {"candidate", "probe"}
        assert row["mismatches"] == []
        assert all(call["matches_frozen"] for call in row["calls"])
        aa = [call for call in row["calls"] if call["stage"] == "aa"]
        assert all("phases" not in call and "gc_events" not in call for call in aa)
        phase = [call for call in row["calls"] if call["stage"] == "phase"]
        assert all("phases" in call and "gc_events" in call for call in phase)


def test_changed_actual_work_is_retained_and_charged(synthetic):
    synthetic.outputs["candidate"]["paid_work"] = 7
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    assert record["work"]["candidate"] == {"calls": 460, "paid_work": 3220}
    assert record["work"]["probe"] == {"calls": 16, "paid_work": 16}
    mismatches = [item for row in record["states"] for item in row["mismatches"]]
    assert len(mismatches) == 460
    assert all(item["implementation"] == "candidate" and item["result"]["paid_work"] == 7
               for item in mismatches)
    assert sum(not call["matches_frozen"] for row in record["states"] for call in row["calls"]) == 460


@pytest.mark.parametrize("invalid", ["reference", "screen_metric", "inexact"])
def test_reference_quality_and_exactness_remain_required(synthetic, invalid):
    if invalid == "reference":
        synthetic.states[0]["effective_reference_complete"] = False
    elif invalid == "screen_metric":
        synthetic.states[0]["screen_metrics"]["candidate"]["canonical_action_mismatch_count"] = 1
    else:
        synthetic.outputs["candidate"]["selection"]["canonical_exact"] = False
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    assert record["states"][0]["mismatches"]
    assert any(not call["matches_frozen"] for call in record["states"][0]["calls"])
