from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_large_r as screen
from algorithm.experiments import integer_capacity_large_r_summary as summary
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@pytest.fixture
def model(tmp_path, monkeypatch):
    model = FrameModel(
        node="large_r_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(
            FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
            FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
            FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ())),
        ),
    )
    source = tmp_path / "selector.py"
    source.write_text("frozen source\n")
    previous = tmp_path / "prior_protocol.json"
    previous.write_text(json.dumps({
        "models": [model.snapshot()], "execution_sources": {"selector.py": "frozen source\n"},
        "search_counters": runtime.COUNTERS, "capacity_counters": runtime.CAPACITY_COUNTERS,
    }))
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({
        "decision": "UTILITY_REVIEW_REQUIRED", "integration_gate_passed": True,
        "utility_gate_passed": False, "flow_gain_established": False,
    }))
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "PRIOR_PROTOCOL", previous)
    monkeypatch.setattr(screen, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(screen, "EXTRA_SOURCES", ())
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    monkeypatch.setattr(screen, "runtime_setup", lambda: runtime)
    return model


def tiny_protocol(model, lanes=32):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[lanes], registered_at_utc="test")
    protocol["queue_panels"] = {
        str(lanes): [{"panel": "test", "states": [{"state_id": "test:0", "queue": [3, 3, 0]}]}],
    }
    return protocol


@pytest.fixture
def tiny_result(model):
    protocol = tiny_protocol(model)
    return protocol, screen.screen_cell(model, 32, protocol, progress=False)


def scaled_cell(protocol, row):
    return row["states"][0]["budgets"][str(protocol["budget_regimes"]["scaled_36r"][str(row["lanes"])])]


def set_cost(record, value):
    for counter in ("lp_calls", "lp_matrix_coefficients"):
        record["metrics"]["search_work"][counter] = value
        record["selection"][counter] = value


def make_costs_pass(protocol, row):
    cell = scaled_cell(protocol, row)
    for label in ("old_shared", "fractional_shared"):
        set_cost(cell[label], 100)
    set_cost(cell["integer_capacity"], 10)


def make_unresolved(record, *, cap=False):
    selection, metrics = record["selection"], record["metrics"]
    selection.update(exact=False, score_exact=False, canonical_exact=False,
                     upper_bound=selection["score"] + 1, gap=1.0)
    metrics.update(score_exact_count=0, canonical_exact_count=0,
                   gap_sum=1.0, max_gap=1.0)
    if cap:
        selection["lp_calls"] = selection["lp_node_budget"]
        metrics["search_work"]["lp_calls"] = selection["lp_node_budget"]
        metrics.update(cap_reached_count=1, uncertified_cap_reached_count=1)


def test_frozen_matrix_and_coupled_dyadic_queues_are_deterministic(model):
    protocol = screen.build_protocol((model,))
    assert protocol == screen.build_protocol((model,))
    assert protocol["lanes"] == [16, 32, 64]
    assert protocol["reference_budget"] == 4096
    assert protocol["unique_budgets"] == {"16": [576], "32": [576, 1152], "64": [576, 2304]}
    assert protocol["budget_regimes"] == {
        "fixed_576": {"16": 576, "32": 576, "64": 576},
        "scaled_36r": {"16": 576, "32": 1152, "64": 2304},
    }
    states = {}
    for lanes in (16, 32, 64):
        rows = [state for panel in protocol["queue_panels"][str(lanes)] for state in panel["states"]]
        assert len(rows) == 76
        assert len({state["state_id"] for state in rows}) == len(rows)
        assert all(len(state["queue"]) == 3 and all(isinstance(value, int) and value >= 0 for value in state["queue"]) for state in rows)
        states[lanes] = {state["state_id"]: state["queue"] for state in rows}
    assert set(states[16]) == set(states[32]) == set(states[64])
    random_ids = [state["state_id"] for panel in protocol["queue_panels"]["16"] if panel["panel"] != "boundary" for state in panel["states"]]
    assert len(random_ids) == 64
    for state_id in random_ids:
        assert states[16][state_id] == [value // 2 for value in states[32][state_id]]
        assert states[32][state_id] == [value // 2 for value in states[64][state_id]]
    assert any(value % 2 for state_id in random_ids for value in states[32][state_id])
    assert any(value % 4 for state_id in random_ids for value in states[64][state_id])


@pytest.mark.parametrize("lanes, budgets", [(16, [576]), (32, [576, 1152]), (64, [576, 2304])])
def test_reference_once_and_selectors_once_per_distinct_budget(model, monkeypatch, lanes, budgets):
    protocol = tiny_protocol(model, lanes)
    calls = []
    for name, label in (
        ("select_canonical_count_space_bnb_frame", "reference"),
        ("select_shared_threshold_canonical_requirement_dominance_frame", "old_shared"),
        ("select_fractional_shared_threshold_canonical_requirement_dominance_frame", "fractional_shared"),
        ("select_integer_capacity_shared_threshold_frame", "integer_capacity"),
    ):
        original = getattr(runtime, name)

        def tracked(options, queue, actual_lanes, budget, selector=original, kind=label):
            calls.append((kind, tuple(queue), actual_lanes, budget))
            return selector(options, queue, actual_lanes, budget)

        monkeypatch.setattr(runtime, name, tracked)
    row = screen.screen_cell(model, lanes, protocol, progress=False)
    assert calls == [("reference", (3, 3, 0), lanes, 4096)] + [
        (label, (3, 3, 0), lanes, budget) for budget in budgets
        for label in ("old_shared", "fractional_shared", "integer_capacity")
    ]
    result = summary.summarize((model,), protocol, [row])
    assert result["executed_selector_calls"] == 1 + 3 * len(budgets)
    assert result["references_complete"] and result["scaled_action_gate_passed"]


def test_main_scaled_gate_is_independent_of_honest_fixed_cap_nonclosure(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    make_costs_pass(protocol, row)
    fixed = row["states"][0]["budgets"]["576"]["integer_capacity"]
    make_unresolved(fixed, cap=True)
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INTEGER_CAPACITY_LARGE_R_SCALING_ADVANCE"
    assert result["scaled_action_gate_passed"] and result["scaled_cost_gate_passed"]
    assert not result["regimes"]["fixed_576"]["action_gate_passed"]
    fixed_metrics = result["regimes"]["fixed_576"]["metrics"]["integer_capacity"]
    assert fixed_metrics["canonical_exact_count"] == 0
    assert fixed_metrics["uncertified_cap_reached_count"] == 1
    assert fixed_metrics["failures"] == 0


@pytest.mark.parametrize("control", ["old_shared", "fractional_shared"])
def test_one_node_cost_regression_cannot_hide_in_pooled_savings(model, tiny_result, control):
    protocol, row = deepcopy(tiny_result)
    other_model = replace(model, node="other")
    rows = [row, deepcopy(row)]
    rows[1]["node"] = other_model.node
    for item, old, new in ((rows[0], 10, 11), (rows[1], 100, 1)):
        make_costs_pass(protocol, item)
        cell = scaled_cell(protocol, item)
        set_cost(cell[control], old)
        set_cost(cell["integer_capacity"], new)
    result = summary.summarize((model, other_model), protocol, rows)
    assert result["scaled_action_gate_passed"]
    assert not result["scaled_cost_gate_passed"]
    assert not result["regimes"]["scaled_36r"]["cost_gate_by_control"][control]
    assert result["decision"] == "LARGE_R_SEARCH_COST_REVIEW_REQUIRED"


@pytest.mark.parametrize("failure", ["certificate", "unresolved", "mismatch", "reference"])
def test_incomplete_or_invalid_outcomes_are_preserved_and_cannot_advance(model, tiny_result, failure):
    protocol, row = deepcopy(tiny_result)
    make_costs_pass(protocol, row)
    state = row["states"][0]
    integer = scaled_cell(protocol, row)["integer_capacity"]
    if failure == "certificate":
        integer["metrics"]["failures"] = 1
    elif failure == "unresolved":
        make_unresolved(integer, cap=True)
    elif failure == "mismatch":
        integer["metrics"]["canonical_action_mismatch_count"] = 1
    else:
        state["reference_complete"] = False
        make_unresolved({"selection": state["reference"], "metrics": state["reference_metrics"]})
        for candidates in state["budgets"].values():
            for record in candidates.values():
                record["metrics"]["reference_comparable_count"] = 0
    result = summary.summarize((model,), protocol, [row])
    assert not result["scaled_action_gate_passed"]
    assert not result["decision"].endswith("ADVANCE")
    if failure == "reference":
        assert result["decision"] == "INCOMPLETE_REFERENCE"
        assert not result["references_complete"]
    elif failure == "certificate":
        assert result["decision"] == "INVALID_EXECUTION"
    else:
        assert result["decision"] == "LARGE_R_ACTION_REVIEW_REQUIRED"


@pytest.mark.parametrize("change", ["missing_cell", "duplicate_cell", "state_id", "queue", "budget", "calls", "exact_flag"])
def test_summary_rejects_lost_coverage_or_duplicated_work(model, tiny_result, change):
    protocol, row = deepcopy(tiny_result)
    rows = [row]
    if change == "missing_cell":
        rows = []
    elif change == "duplicate_cell":
        rows.append(deepcopy(row))
    elif change == "state_id":
        row["states"][0]["state_id"] = "unregistered"
    elif change == "queue":
        row["states"][0]["queue"] = [4, 3, 0]
    elif change == "budget":
        row["states"][0]["budgets"].pop("576")
    elif change == "calls":
        scaled_cell(protocol, row)["integer_capacity"]["metrics"]["calls"] = 2
    else:
        scaled_cell(protocol, row)["integer_capacity"]["selection"]["canonical_exact"] = False
    with pytest.raises(ValueError):
        summary.summarize((model,), protocol, rows)


def test_incomplete_reference_is_kept_without_extra_budget_or_candidate_rescue(model, monkeypatch):
    protocol = tiny_protocol(model)
    original = runtime.select_canonical_count_space_bnb_frame
    budgets = []

    def unresolved(options, queue, lanes, budget):
        budgets.append(budget)
        selected = original(options, queue, lanes, budget)
        return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=selected.score + 1, gap=1.0, epsilon=1.0 / sum(queue))

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", unresolved)
    row = screen.screen_cell(model, 32, protocol, progress=False)
    state, = row["states"]
    assert budgets == [4096]
    assert not state["reference_complete"]
    assert state["reference_metrics"]["failures"] == 0
    for records in state["budgets"].values():
        for record in records.values():
            assert record["metrics"]["reference_comparable_count"] == 0
            assert record["metrics"]["canonical_exact_count"] == 1
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["first_incomplete_reference"]["state_id"] == "test:0"
    assert not result["scaled_action_gate_passed"]


@pytest.mark.parametrize("change", ["source", "model", "lanes", "queue", "budget"])
def test_registered_source_model_and_matrix_mutations_stop_before_execution(model, tmp_path, monkeypatch, change):
    output, source = tmp_path / "large_r.json", tmp_path / "selector.py"
    source.write_text("frozen source\n")
    monkeypatch.setattr(runtime, "load_catalogues", lambda: (model,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    path = output.with_name(output.stem + "_protocol.json")
    protocol = json.loads(path.read_text())
    if change == "source":
        source.write_text("changed source\n")
    elif change == "model":
        monkeypatch.setattr(runtime, "load_catalogues", lambda: (replace(model, lanes=3),))
    elif change == "lanes":
        protocol["lanes"].append(128)
    elif change == "queue":
        protocol["queue_panels"]["16"][0]["states"][0]["queue"][0] += 1
    else:
        protocol["budget_regimes"]["scaled_36r"]["32"] += 1
    path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("changed protocol reached selector calls")

    monkeypatch.setattr(screen, "screen_cell", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--run"])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()
