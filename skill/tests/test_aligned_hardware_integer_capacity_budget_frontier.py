from copy import deepcopy
from dataclasses import replace
from itertools import product
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_budget_frontier as screen
from algorithm.experiments import aligned_hardware_integer_capacity_large_r as previous_screen
from algorithm.experiments import integer_capacity_budget_frontier_summary as summary
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@pytest.fixture
def model(tmp_path, monkeypatch):
    model = FrameModel(
        node="frontier_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(
            FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
            FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
            FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ())),
        ),
    )
    (tmp_path / "selector.py").write_text("frozen source\n")
    previous = tmp_path / "prior_protocol.json"
    previous.write_text(json.dumps({
        "models": [model.snapshot()], "execution_sources": {"selector.py": "frozen source\n"},
        "search_counters": runtime.COUNTERS, "capacity_counters": runtime.CAPACITY_COUNTERS,
        "queue_panels": previous_screen.queue_design()[0],
    }))
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({
        "decision": "INTEGER_CAPACITY_LARGE_R_SCALING_ADVANCE",
        "certificate_checks_passed": True, "references_complete": True,
        "scaled_action_gate_passed": True, "scaled_cost_gate_passed": True,
    }))
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "PRIOR_PROTOCOL", previous)
    monkeypatch.setattr(screen, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(screen, "EXTRA_SOURCES", ())
    monkeypatch.setattr(previous_screen, "runtime_provenance", lambda: {})
    monkeypatch.setattr(screen, "runtime_setup", lambda: runtime)
    return model


def tiny_protocol(model, lanes=8):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[lanes], registered_at_utc="test", retained_witnesses=[])
    protocol["queue_panels"] = {
        str(lanes): [{"panel": "test", "states": [{"state_id": "test:0", "queue": [3, 3, 0]}]}],
    }
    return protocol


@pytest.fixture
def tiny_result(model):
    protocol = tiny_protocol(model)
    return protocol, screen.screen_cell(model, 8, protocol, progress=False)


def record(row, budget, label="integer_capacity"):
    return row["states"][0]["budgets"][str(budget)][label]


def make_unresolved(item, *, zero_gap=False):
    selection, metrics = item["selection"], item["metrics"]
    selection.update(exact=False, score_exact=zero_gap, canonical_exact=False,
                     upper_bound=selection["score"] + (0 if zero_gap else 1),
                     gap=0.0 if zero_gap else 1.0)
    metrics.update(score_exact_count=int(zero_gap), canonical_exact_count=0,
                   gap_sum=selection["gap"], max_gap=selection["gap"])


def test_roster_preserves_old_queues_and_all_three_hard_neighborhoods(model):
    protocol = screen.build_protocol((model,))
    assert protocol == screen.build_protocol((model,))
    assert protocol["lanes"] == [8, 16, 32, 64]
    assert protocol["budget_grid"] == [9, 18, 36, 72, 144, 288, 576]
    assert protocol["reference_budget"] == 4096
    assert all(caps == protocol["budget_grid"] for caps in protocol["unique_budgets"].values())
    old_panels = previous_screen.queue_design()[0]
    roster = {}
    for lanes, count in ((8, 54), (16, 103), (32, 76), (64, 76)):
        panels = protocol["queue_panels"][str(lanes)]
        states = [state for panel in panels for state in panel["states"]]
        roster[lanes] = {tuple(state["queue"]) for state in states}
        assert len(states) == len(roster[lanes]) == count
        assert len({state["state_id"] for state in states}) == count
        if lanes != 8:
            assert panels[:len(old_panels[str(lanes)])] == json.loads(json.dumps(old_panels[str(lanes)]))
    assert sum(map(len, roster.values())) * 4 == 1236
    assert len(protocol["retained_witnesses"]) == 3
    for witness in protocol["retained_witnesses"]:
        neighbors = {tuple(value + delta for value, delta in zip(witness["queue"], offset))
                     for offset in product((-1, 0, 1), repeat=3)}
        assert neighbors <= roster[witness["lanes"]]


def test_reference_once_and_each_budget_is_a_separate_cold_call(model, monkeypatch):
    protocol = tiny_protocol(model)
    calls = []
    for name, label in (
        ("select_canonical_count_space_bnb_frame", "reference"),
        ("select_shared_threshold_canonical_requirement_dominance_frame", "old_shared"),
        ("select_fractional_shared_threshold_canonical_requirement_dominance_frame", "fractional_shared"),
        ("select_integer_capacity_shared_threshold_frame", "integer_capacity"),
    ):
        original = getattr(runtime, name)

        def tracked(options, queue, lanes, budget, selector=original, kind=label):
            calls.append((kind, tuple(queue), lanes, budget))
            return selector(options, queue, lanes, budget)

        monkeypatch.setattr(runtime, name, tracked)
    row = screen.screen_cell(model, 8, protocol, progress=False)
    assert calls == [("reference", (3, 3, 0), 8, 4096)] + [
        (label, (3, 3, 0), 8, cap) for cap in protocol["budget_grid"]
        for label in ("old_shared", "fractional_shared", "integer_capacity")
    ]
    result = summary.summarize((model,), protocol, [row])
    assert result["executed_reference_calls"] == 1
    assert result["executed_candidate_calls"] == 21
    assert result["executed_selector_calls"] == 22
    assert result["decision"] == "INTEGER_CAPACITY_BUDGET_FRONTIER_CHARACTERIZED"


def test_correct_action_and_zero_gap_do_not_imply_canonical_closure(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    make_unresolved(record(row, 9), zero_gap=True)
    result = summary.summarize((model,), protocol, [row])
    curve = result["curves"][0]
    rates = curve["label_rates"]["integer_capacity"]
    assert rates["observed_score_equal_count"] == 1
    assert rates["observed_canonical_action_equal_count"] == 1
    assert rates["score_exact_count"] == 1
    assert rates["canonical_exact_count"] == rates["certified_canonical_action_equal_count"] == 0
    assert curve["normalized_gaps"]["integer_capacity"]["zero_gap_uncertified_count"] == 1
    assert result["budget_boundaries"]["integer_capacity"] == {
        "observed_canonical_action": 9, "certified_canonical": 18,
    }


def test_frontier_uses_all_higher_tested_caps_and_reports_lost_closure(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    make_unresolved(record(row, 18))
    record(row, 18)["metrics"]["canonical_action_mismatch_count"] = 1
    result = summary.summarize((model,), protocol, [row])
    assert result["budget_boundaries"]["integer_capacity"] == {
        "observed_canonical_action": 36, "certified_canonical": 36,
    }
    regressions = result["budget_regressions"]["integer_capacity"]
    for name in ("score_certificate_lost", "canonical_certificate_lost", "observed_action_lost",
                 "upper_bound_increased"):
        assert regressions[name]["count"] == 1
        assert regressions[name]["first"]["lower_budget"] == 9
        assert regressions[name]["first"]["higher_budget"] == 18


def test_observed_canonical_action_requires_score_agreement_too(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    make_unresolved(record(row, 9))
    record(row, 9)["metrics"]["score_mismatch_count"] = 1
    result = summary.summarize((model,), protocol, [row])
    rates = result["curves"][0]["label_rates"]["integer_capacity"]
    assert rates["observed_score_equal_count"] == 0
    assert rates["observed_canonical_action_equal_count"] == 0
    assert result["budget_boundaries"]["integer_capacity"]["observed_canonical_action"] == 18


def test_honest_high_cap_nonclosure_is_hold_without_becoming_invalid(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    make_unresolved(record(row, 576), zero_gap=True)
    result = summary.summarize((model,), protocol, [row])
    assert result["certificate_checks_passed"] and result["references_complete"]
    assert result["decision"] == "BUDGET_FRONTIER_HIGH_CAP_HOLD"
    assert not result["high_cap_action_gate_passed"]
    assert result["budget_boundaries"]["integer_capacity"]["certified_canonical"] is None
    assert result["budget_boundaries"]["integer_capacity"]["observed_canonical_action"] == 9


def test_invalid_low_cap_cannot_be_hidden_by_successful_high_cap(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    record(row, 9, "old_shared")["metrics"]["failures"] = 1
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"]
    assert not result["high_cap_action_gate_passed"]
    assert all(value is None for boundaries in result["budget_boundaries"].values()
               for value in boundaries.values())


def test_incomplete_reference_cannot_be_rescued_by_candidate_closure(model, monkeypatch):
    protocol = tiny_protocol(model)
    original = runtime.select_canonical_count_space_bnb_frame
    budgets = []

    def unresolved(options, queue, lanes, budget):
        budgets.append(budget)
        selected = original(options, queue, lanes, budget)
        return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=selected.score + 1, gap=1.0, epsilon=1.0 / sum(queue))

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", unresolved)
    row = screen.screen_cell(model, 8, protocol, progress=False)
    assert budgets == [4096]
    for selected in row["states"][0]["budgets"].values():
        assert selected["integer_capacity"]["metrics"]["canonical_exact_count"] == 1
        assert selected["integer_capacity"]["metrics"]["reference_comparable_count"] == 0
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert result["first_incomplete_reference"]["state_id"] == "test:0"
    assert not result["high_cap_action_gate_passed"]
    assert all(value is None for boundaries in result["budget_boundaries"].values()
               for value in boundaries.values())


@pytest.mark.parametrize("change", ["missing_cell", "duplicate_cell", "missing_state", "duplicate_state",
                                   "queue", "budget", "label", "calls", "exact_flag", "call_budget"])
def test_summary_rejects_lost_coverage_or_duplicated_work(model, tiny_result, change):
    protocol, row = deepcopy(tiny_result)
    rows = [row]
    if change == "missing_cell":
        rows = []
    elif change == "duplicate_cell":
        rows.append(deepcopy(row))
    elif change == "missing_state":
        row["states"].clear()
    elif change == "duplicate_state":
        row["states"].append(deepcopy(row["states"][0]))
    elif change == "queue":
        row["states"][0]["queue"] = [4, 3, 0]
    elif change == "budget":
        row["states"][0]["budgets"].pop("9")
    elif change == "label":
        row["states"][0]["budgets"]["9"].pop("old_shared")
    elif change == "calls":
        record(row, 9)["metrics"]["calls"] = 2
    elif change == "exact_flag":
        record(row, 9)["selection"]["canonical_exact"] = False
    else:
        record(row, 9)["selection"]["lp_node_budget"] = 18
    with pytest.raises(ValueError):
        summary.summarize((model,), protocol, rows)


@pytest.mark.parametrize("change", ["source", "model", "lanes", "queue", "budget"])
def test_frozen_source_model_and_matrix_mutations_stop_before_execution(model, tmp_path, monkeypatch, change):
    output, source = tmp_path / "frontier.json", tmp_path / "selector.py"
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
        protocol["queue_panels"]["8"][0]["states"][0]["queue"][0] += 1
    else:
        protocol["budget_grid"][0] += 1
    path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("changed protocol reached selector calls")

    monkeypatch.setattr(screen, "screen_cell", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--run"])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()
