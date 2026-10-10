from collections import Counter
from copy import deepcopy
from dataclasses import replace
import gzip
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as confirmation
from algorithm.experiments import aligned_hardware_strict_score_proof_timing as timing
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.strict_score_proof_canonical_completion_frame_candidates import (
    select_strict_score_proof_canonical_completion_frame,
)


@pytest.fixture
def observed(monkeypatch):
    monkeypatch.setattr(confirmation, "runtime_provenance", lambda: {})
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {})
    model = FrameModel(
        node="strict_timing_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )
    protocol = {
        "registered_at_utc": "synthetic-confirmation", "budget": 144,
        "integer_state_budget": 100000, "reference_budget": 4096,
        "oracle_state_budget": 5000000,
        "queue_panels": {"2": [
            {"panel": "fresh", "seed": 1, "cohort": "fresh_confirmation", "states": [
                {"state_id": "fresh:0", "queue": [3, 3, 0]},
                {"state_id": "fresh:1", "queue": [2, 2, 0]},
            ]},
            {"panel": "regression", "seed": None, "cohort": "old_regression", "states": [
                {"state_id": "regression:0", "queue": [1, 2, 0]},
            ]},
        ]},
    }
    prior = confirmation.screen_cell(model, 2, protocol, runtime, progress=False)
    protocol.update(
        registered_at_utc="synthetic-timing", prior_registered_at_utc="synthetic-confirmation",
        models=[model.snapshot()], lanes=[2], labels=["candidate", "control"],
        lp_budgets={"candidate": 144, "control": 4096}, repeats=6,
        warmups_per_label=2, cpu=0, pooled_ratio_limit=0.95, cell_ratio_limit=1.0,
        search_counters=runtime.COUNTERS, capacity_counters=runtime.CAPACITY_COUNTERS,
        execution_order=[[model.node, 2]],
        timing_orders={f"{model.node}:2": timing.timing_orders(model.node, 2, 3)},
    )
    return model, protocol, prior


def test_registered_orders_balance_each_state_and_cover_every_repeat():
    orders = timing.timing_orders("test", 64, 37)
    assert orders == timing.timing_orders("test", 64, 37)
    assert len(orders["state_orders"]) == 6
    for order in orders["state_orders"]:
        assert sorted(order) == list(range(37))
    assert len(orders["label_orders"]) == 37
    for orders_for_state in orders["label_orders"]:
        assert Counter(tuple(order) for order in orders_for_state) == {
            ("candidate", "control"): 3, ("control", "candidate"): 3,
        }


def test_roster_keeps_fresh_and_regression_metadata(observed):
    _, protocol, _ = observed
    states = timing.state_roster(protocol, 2)
    assert [state["state_id"] for state in states] == ["fresh:0", "fresh:1", "regression:0"]
    assert [state["cohort"] for state in states] == ["fresh_confirmation"] * 2 + ["old_regression"]
    assert [state["seed"] for state in states] == [1, 1, None]
    assert [tuple(state["queue"]) for state in states] == [(3, 3, 0), (2, 2, 0), (1, 2, 0)]


def test_all_calls_are_cold_paired_and_use_the_two_frozen_budgets(observed, monkeypatch):
    model, protocol, prior = observed
    saved_prior = deepcopy(prior)
    calls = []

    def selector(options, queue, lanes, budget, integer_budget):
        calls.append((tuple(queue), lanes, budget, integer_budget))
        return select_strict_score_proof_canonical_completion_frame(
            options, queue, lanes, budget, integer_budget)

    def forbidden(*args, **kwargs):
        raise AssertionError("timing reran a frozen reference")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", forbidden)
    from algorithm.experiments import integer_capacity_canonical_oracle
    monkeypatch.setattr(integer_capacity_canonical_oracle, "canonical_at_certified_score", forbidden)
    row = timing.timing_cell(model, 2, protocol, runtime, prior, selector=selector, progress=False)
    assert len(calls) == 4 + 3 * 6 * 2
    assert Counter(call[2] for call in calls) == {144: 20, 4096: 20}
    assert all(call[1] == 2 and call[3] == 100000 for call in calls)
    assert all(call[0] == (3, 3, 0) for call in calls[:4])
    expected_calls = []
    roster = timing.state_roster(protocol, 2)
    orders = protocol["timing_orders"][f"{model.node}:2"]
    for repeat, indices in enumerate(orders["state_orders"]):
        for index in indices:
            expected_calls.extend(
                (tuple(roster[index]["queue"]), 2, protocol["lp_budgets"][label], 100000)
                for label in orders["label_orders"][index][repeat])
    assert calls[4:] == expected_calls
    assert prior == saved_prior
    assert len(row["states"]) == 3
    for state in row["states"]:
        for label in ("candidate", "control"):
            assert len(state["wall_ns"][label]) == len(state["cpu_ns"][label]) == 6
            assert all(value > 0 for value in state["wall_ns"][label])


def test_measured_intervals_exclude_audits_and_keep_pair_adjacent(observed, monkeypatch):
    model, protocol, prior = observed
    events = []
    clock = {"wall": 0, "cpu": 0}

    def read_clock(name):
        clock[name] += 100
        events.append(name)
        return clock[name]

    monkeypatch.setattr(timing, "perf_counter_ns", lambda: read_clock("wall"))
    monkeypatch.setattr(timing, "process_time_ns", lambda: read_clock("cpu"))
    original_record = runtime._record

    def record(*args, **kwargs):
        events.append("audit")
        return original_record(*args, **kwargs)

    def selector(*args):
        events.append("selector")
        return select_strict_score_proof_canonical_completion_frame(*args)

    monkeypatch.setattr(runtime, "_record", record)
    row = timing.timing_cell(model, 2, protocol, runtime, prior, selector=selector, progress=False)
    first_clock = next(index for index, event in enumerate(events) if event in ("wall", "cpu"))
    measured = events[first_clock:]
    intervals = []
    for index, event in enumerate(measured):
        if event == "selector":
            intervals.append(measured[index - 2:index + 3])
    assert intervals == [["cpu", "wall", "selector", "wall", "cpu"]] * 36
    # All selection validation is deferred until the adjacent two-method block ends.
    selector_count = 0
    for event in measured:
        if event == "selector":
            selector_count += 1
        elif event == "audit":
            assert selector_count % 2 == 0
    assert selector_count == 36
    for state in row["states"]:
        for label in ("candidate", "control"):
            assert state["wall_ns"][label] == [100] * 6
            assert state["cpu_ns"][label] == [100] * 6


def test_nondeterministic_work_is_retained_even_when_action_stays_equal(observed):
    model, protocol, prior = observed
    counts = Counter()

    def selector(options, queue, lanes, budget, integer_budget):
        result = select_strict_score_proof_canonical_completion_frame(
            options, queue, lanes, budget, integer_budget)
        counts[(tuple(queue), budget)] += 1
        if tuple(queue) == (3, 3, 0) and budget == 4096 and counts[(tuple(queue), budget)] == 8:
            result = replace(result, preprocessing=replace(
                result.preprocessing, reduction_option_probes=result.preprocessing.reduction_option_probes + 1))
        return result

    row = timing.timing_cell(model, 2, protocol, runtime, prior, selector=selector, progress=False)
    first = row["states"][0]
    mismatch, = first["repeat_mismatches"]["control"]
    assert mismatch["repeat_index"] == 5
    assert mismatch["result"]["selection"] == first["first_results"]["control"]["selection"]
    assert mismatch["result"]["preprocessing"] != first["first_results"]["control"]["preprocessing"]
    assert all(state["metrics"][label]["calls"] == 6
               for state in row["states"] for label in protocol["labels"])
    assert all(not state["repeat_mismatches"]["candidate"] for state in row["states"])


def test_incomplete_control_keeps_all_repeats_and_later_states(observed):
    model, protocol, prior = observed

    def selector(options, queue, lanes, budget, integer_budget):
        result = select_strict_score_proof_canonical_completion_frame(
            options, queue, lanes, budget, integer_budget)
        if budget == 4096:
            result = replace(result, selection=replace(result.selection, canonical_exact=False, exact=False))
        return result

    row = timing.timing_cell(model, 2, protocol, runtime, prior, selector=selector, progress=False)
    assert len(row["warmups"]) == 4
    assert len(row["states"]) == 3
    for state in row["states"]:
        assert state["metrics"]["candidate"]["canonical_exact_count"] == 6
        assert state["metrics"]["control"]["canonical_exact_count"] == 0
        assert state["metrics"]["control"]["calls"] == 6
        assert len(state["wall_ns"]["control"]) == 6
        assert state["expected_candidate_result"] == timing.confirmed_result(
            next(old for old in prior["states"] if old["state_id"] == state["state_id"]))


@pytest.fixture
def saved_protocol(observed, monkeypatch, tmp_path):
    model, previous, _ = observed
    previous = deepcopy(previous)
    previous["registered_at_utc"] = "synthetic-confirmation"
    previous["execution_sources"] = {"selector.py": "frozen selector\n"}
    (tmp_path / "selector.py").write_text("frozen selector\n")
    prior_protocol = tmp_path / "prior_protocol.json"
    prior_protocol.write_text(json.dumps(previous))
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({
        "decision": "STRICT_SCORE_PROOF_CONFIRMATION_ADVANCE", "certificate_checks_passed": True,
        "effective_references_complete": True, "fresh_confirmation_gate_passed": True,
        "regression_gate_passed": True,
    }))
    for name, value in (("ROOT", tmp_path), ("PRIOR_OUTPUT", prior_output),
                        ("PRIOR_PROTOCOL", prior_protocol), ("EXTRA_SOURCES", ())):
        monkeypatch.setattr(timing, name, value)
    monkeypatch.setattr(timing, "runtime_setup", lambda: runtime)
    monkeypatch.setattr(runtime, "load_catalogues", lambda: (model,))
    output = tmp_path / "timing.json"
    monkeypatch.setattr("sys.argv", ["timing", "--output", str(output), "--register"])
    timing.main()
    return output, output.with_name(output.stem + "_protocol.json")


@pytest.mark.parametrize("change", ("source", "model", "lp_budget", "integer_budget", "order", "queue", "reference_rule"))
def test_changed_protocol_stops_before_any_timed_call(
        observed, saved_protocol, monkeypatch, tmp_path, change):
    model, _, _ = observed
    output, path = saved_protocol
    protocol = json.loads(path.read_text())
    if change == "source":
        (tmp_path / "selector.py").write_text("changed selector\n")
    elif change == "model":
        monkeypatch.setattr(runtime, "load_catalogues", lambda: (replace(model, lanes=3),))
    elif change == "lp_budget":
        protocol["lp_budgets"]["control"] += 1
    elif change == "integer_budget":
        protocol["integer_state_budget"] += 1
    elif change == "order":
        protocol["timing_orders"][f"{model.node}:2"]["label_orders"][0][0].reverse()
    elif change == "queue":
        protocol["queue_panels"]["2"][0]["states"][0]["queue"][0] += 1
    else:
        protocol["reference_policy"] = "Use the candidate as its own reference"
    path.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        raise AssertionError("changed protocol reached experiment calls")

    monkeypatch.setattr(timing, "timing_cell", forbidden)
    monkeypatch.setattr("sys.argv", ["timing", "--output", str(output), "--run"])
    with pytest.raises(ValueError):
        timing.main()
    assert not output.exists()


@pytest.mark.parametrize("change", (None, "registration", "queue", "cohort"))
def test_reused_cell_must_match_confirmation_and_frozen_roster(observed, monkeypatch, tmp_path, change):
    model, protocol, prior = observed
    protocol["prior_result_path"] = "prior.json"
    monkeypatch.setattr(timing, "ROOT", tmp_path)
    saved = {"registered_at_utc": "synthetic-confirmation", "cell": deepcopy(prior)}
    if change == "registration":
        saved["registered_at_utc"] = "other-confirmation"
    elif change == "queue":
        saved["cell"]["states"][0]["queue"][0] += 1
    elif change == "cohort":
        saved["cell"]["states"][0]["cohort"] = "old_regression"
    path = timing._cell_path(tmp_path / "prior.json", model.node, 2)
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(saved, target)
    if change is None:
        actual = timing.load_prior_cell(protocol, model.node, 2)
        assert actual == json.loads(json.dumps(prior))
    else:
        with pytest.raises(ValueError):
            timing.load_prior_cell(protocol, model.node, 2)
