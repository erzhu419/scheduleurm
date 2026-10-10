"""Diagnostic timing boundaries, balanced order, and transparent failures."""
from collections import Counter
from copy import deepcopy
import gc
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_resumable_probe_diagnosis as diagnosis


def test_each_queue_has_balanced_gc_and_selector_orders():
    orders = diagnosis.timing_orders(count=7, repeats=12, seed=123)
    assert orders == diagnosis.timing_orders(count=7, repeats=12, seed=123)
    assert len(orders["state_orders"]) == 12
    assert all(sorted(order) == list(range(7)) for order in orders["state_orders"])
    assert len(orders["condition_orders"]) == len(orders["label_orders"]) == 7
    for index in range(7):
        assert Counter(map(tuple, orders["condition_orders"][index])) == {
            ("enabled", "disabled"): 6, ("disabled", "enabled"): 6,
        }
        for condition in ("enabled", "disabled"):
            assert Counter(map(tuple, orders["label_orders"][index][condition])) == {
                ("candidate", "legacy"): 6, ("legacy", "candidate"): 6,
            }


@pytest.fixture
def gc_state():
    enabled = gc.isenabled()
    callbacks = list(gc.callbacks)
    yield enabled, callbacks
    if enabled:
        gc.enable()
    else:
        gc.disable()
    gc.callbacks[:] = callbacks


def test_capture_covers_entire_callable_and_excludes_later_validation(monkeypatch):
    events = []
    counts = {"wall": 0, "cpu": 0}

    def clock(kind):
        events.append(kind)
        counts[kind] += 100
        return counts[kind]

    monkeypatch.setattr(diagnosis, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(diagnosis, "process_time_ns", lambda: clock("cpu"))

    def selector():
        events.append("selector")
        return {"action": 7}

    with diagnosis.Instrumentation() as instrument:
        output, measured = instrument.capture("candidate", selector)
    events.append("audit")
    assert output == {"action": 7}
    selector_index = events.index("selector")
    assert set(events[selector_index - 2:selector_index]) == {"cpu", "wall"}
    assert set(events[selector_index + 1:selector_index + 3]) == {"cpu", "wall"}
    assert events[-1] == "audit"
    assert measured["wall_ns"] == measured["cpu_ns"] == 100


@pytest.mark.parametrize("original_enabled", [True, False])
@pytest.mark.parametrize("condition", ["enabled", "disabled"])
def test_pair_collects_before_adjacent_calls_and_restores_gc(
    monkeypatch, gc_state, original_enabled, condition,
):
    (gc.enable if original_enabled else gc.disable)()
    events = []

    def collect():
        events.append("collect")
        return 7

    def capture(label, selector):
        events.append(("capture", label, gc.isenabled()))
        return selector(), {"wall_ns": 100, "cpu_ns": 90}

    monkeypatch.setattr(diagnosis.gc, "collect", collect)
    instrument = SimpleNamespace(capture=capture)
    outputs, record = diagnosis.run_pair(
        instrument, condition, ["legacy", "candidate"],
        {"candidate": lambda: "candidate-result", "legacy": lambda: "legacy-result"},
    )
    assert outputs == {"legacy": "legacy-result", "candidate": "candidate-result"}
    assert events == ["collect", ("capture", "legacy", condition == "enabled"),
                      ("capture", "candidate", condition == "enabled")]
    assert gc.isenabled() == original_enabled
    assert record is not None


@pytest.mark.parametrize("original_enabled", [True, False])
def test_pair_restores_gc_if_selector_raises(monkeypatch, gc_state, original_enabled):
    (gc.enable if original_enabled else gc.disable)()
    monkeypatch.setattr(diagnosis.gc, "collect", lambda: 0)

    def fail():
        raise RuntimeError("selector failed")

    instrument = SimpleNamespace(capture=lambda label, call: (call(), {}))
    with pytest.raises(RuntimeError, match="selector failed"):
        diagnosis.run_pair(instrument, "disabled" if original_enabled else "enabled",
                           ["candidate", "legacy"], {"candidate": fail, "legacy": lambda: None})
    assert gc.isenabled() == original_enabled


@pytest.mark.parametrize("fail", [False, True])
def test_instrumentation_restores_function_patches_and_gc_callbacks(gc_state, fail):
    from algorithm import resumable_probe_frame_candidates as candidate
    from algorithm import resumable_integer_capacity_frame_candidates as engine
    from algorithm import strict_score_proof_canonical_completion_frame_candidates as legacy

    owners = [candidate, engine, legacy, engine.IntegerCapacitySearch]
    previous = {owner: {name: value for name, value in vars(owner).items() if callable(value)}
                for owner in owners}
    callbacks = list(gc.callbacks)

    def run():
        with diagnosis.Instrumentation():
            assert any(getattr(owner, name) is not value
                       for owner, items in previous.items() for name, value in items.items())
            assert len(gc.callbacks) == len(callbacks) + 1
            if fail:
                raise RuntimeError("diagnostic failed")

    if fail:
        with pytest.raises(RuntimeError, match="diagnostic failed"):
            run()
    else:
        run()
    assert gc.callbacks == callbacks
    for owner, items in previous.items():
        assert all(getattr(owner, name) is value for name, value in items.items())


def test_nested_lp_completion_and_gc_times_are_not_charged_twice(monkeypatch, gc_state):
    gc.disable()
    now = [0]
    monkeypatch.setattr(diagnosis, "perf_counter_ns", lambda: now[0])
    monkeypatch.setattr(diagnosis, "process_time_ns", lambda: now[0])
    with diagnosis.Instrumentation() as instrument:
        def lp():
            now[0] += 10
            instrument._gc_callback("start", {"generation": 0})
            now[0] += 3
            instrument._gc_callback("stop", {"generation": 0, "collected": 1, "uncollectable": 0})
            now[0] += 7

        nested_lp = instrument._wrap(lp, "linprog", False)

        def initial():
            now[0] += 5
            nested_lp()
            now[0] += 5

        initial_lp = instrument._wrap(initial, "initial_lp", True)

        def completion():
            now[0] += 4

        nested_completion = instrument._wrap(completion, "completion", False)

        def probe():
            now[0] += 3
            nested_completion()
            now[0] += 3

        integer_probe = instrument._wrap(probe, "probe", True)

        def selector():
            initial_lp()
            integer_probe()
            now[0] += 2
            return "selected"

        _, measured = instrument.capture("candidate", selector)
    for clock in ("wall_ns", "cpu_ns"):
        assert measured[clock] == 42
        assert measured["phases"]["initial_lp"][clock] == 30
        assert measured["phases"]["linprog"][clock] == 20
        assert measured["phases"]["probe"][clock] == 10
        assert measured["phases"]["completion"][clock] == 4
        assert measured["primary_phase_" + clock] == 40
        assert measured["residual_" + clock] == 2
        assert measured["gc_events"][0][clock] == 3
    assert measured["gc_events"][0]["phase"] == "linprog"


@pytest.fixture
def synthetic(monkeypatch):
    from algorithm import resumable_probe_frame_candidates as candidate
    from algorithm import strict_score_proof_canonical_completion_frame_candidates as legacy
    from algorithm.experiments import resumable_probe_development_summary as development_summary

    events = []
    selection = {"exact": True, "score_exact": True, "canonical_exact": True}
    expected = {label: {"selection": dict(selection), "arm": label} for label in diagnosis.LABELS}
    outputs = deepcopy(expected)
    metrics = {"failures": 0, "score_mismatch_count": 0, "canonical_action_mismatch_count": 0}
    state = {"name": "synthetic", "node": "mock", "lanes": 2, "queue": [1, 1],
             "expected_results": expected, "effective_reference_complete": True,
             "screen_metrics": {label: dict(metrics) for label in diagnosis.LABELS}}
    protocol = {"states": [state], "repeats": 2, "warmups_per_label_condition_state": 2,
                "timing_orders": diagnosis.timing_orders(count=1, repeats=2, seed=7),
                "warmup_orders": diagnosis.timing_orders(count=1, repeats=2, seed=8),
                "cpu": 0, "prior_quality_passed": True, "limitations": "Mock diagnosis."}

    def select(label):
        def call(*args):
            events.append(("selector", label))
            return deepcopy(outputs[label])
        return call

    def serialize(result):
        events.append(("serialize", result["arm"]))
        return deepcopy(result)

    original_validate = diagnosis.validate_result

    def validate(*args):
        events.append(("validate", args[0]["arm"]))
        return original_validate(*args)

    monkeypatch.setattr(candidate, "select_resumable_probe_frame", select("candidate"))
    monkeypatch.setattr(legacy, "select_strict_score_proof_canonical_completion_frame", select("legacy"))
    monkeypatch.setattr(diagnosis, "asdict", serialize)
    monkeypatch.setattr(diagnosis, "validate_result", validate)
    monkeypatch.setattr(diagnosis, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    monkeypatch.setattr(diagnosis.gc, "collect", lambda: 0)
    monkeypatch.setattr(development_summary, "_candidate_work", lambda values: {"calls": len(values)})
    monkeypatch.setattr(development_summary, "_strict_work", lambda values: {"calls": len(values)})
    return SimpleNamespace(protocol=protocol, state=state, outputs=outputs, events=events,
                           models=[SimpleNamespace(node="mock", options=())])


def test_serialization_and_validation_follow_both_calls_in_every_pair(synthetic):
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    calls = 0
    for kind, _ in synthetic.events:
        if kind == "selector":
            calls += 1
        else:
            assert calls > 0 and calls % 2 == 0
    assert calls == 16
    assert len(record["states"][0]["pairs"]) == 8
    assert len(record["states"][0]["calls"]) == 16
    result = diagnosis.summarize(synthetic.protocol, record)
    assert result["decision"] == "DEVELOPMENT_DIAGNOSIS_COMPLETE"
    assert result["execution_checks_passed"]
    assert result["timed_calls"] == result["warmup_calls"] == 8
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0


def test_changed_result_is_retained_and_makes_diagnosis_invalid(synthetic):
    synthetic.outputs["candidate"]["changed_work"] = 1
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    row = record["states"][0]
    assert len(row["calls"]) == 16
    assert len(row["mismatches"]) == 8
    assert all(item["result"]["changed_work"] == 1 for item in row["mismatches"])
    result = diagnosis.summarize(synthetic.protocol, record)
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["execution_checks_passed"]


@pytest.mark.parametrize("failure", ["prior_quality", "incomplete_reference", "failed_screen"])
def test_borrowed_quality_failure_cannot_be_reported_as_valid(synthetic, failure):
    if failure == "prior_quality":
        synthetic.protocol["prior_quality_passed"] = False
    elif failure == "incomplete_reference":
        synthetic.state["effective_reference_complete"] = False
    else:
        synthetic.state["screen_metrics"]["legacy"]["canonical_action_mismatch_count"] = 1
    record = diagnosis.run_diagnosis(synthetic.protocol, synthetic.models, progress=False)
    result = diagnosis.summarize(synthetic.protocol, record)
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["execution_checks_passed"]
    assert result["executed_selector_calls"] == 16


@pytest.mark.parametrize("certificate", ["exact", "score_exact", "canonical_exact"])
def test_identical_but_incomplete_selection_is_not_valid(certificate):
    expected = {"selection": {"exact": True, "score_exact": True, "canonical_exact": True}}
    expected["selection"][certificate] = False
    metrics = {"failures": 0, "score_mismatch_count": 0, "canonical_action_mismatch_count": 0}
    assert not diagnosis.validate_result(deepcopy(expected), expected, True, metrics)
