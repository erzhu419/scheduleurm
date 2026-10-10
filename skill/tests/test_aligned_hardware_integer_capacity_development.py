from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import aligned_hardware_integer_capacity_development as screen


@pytest.fixture
def model():
    return FrameModel(
        node="capacity_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(
            FrameOption("a_early", (1, 0, 0), 3.0, ((3.0,), (), ())),
            FrameOption("a_late", (1, 0, 0), 1.0, ((1.0,), (), ())),
            FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ())),
        ),
    )


def tiny_protocol(model):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[4], registered_at_utc="test")
    protocol["queue_panels"] = {"4": [{"panel": "development", "queues": [[3, 3, 0], [2, 4, 1]]}]}
    return protocol


@pytest.fixture
def tiny_result(model):
    protocol = tiny_protocol(model)
    return protocol, screen.screen_cell(model, 4, protocol, progress=False)


def test_roster_and_budgets_are_the_literal_frozen_development_inputs(model):
    protocol = screen.build_protocol((model,))
    previous = json.loads(screen.PRIOR_DEVELOPMENT.read_text())
    assert protocol["queue_panels"] == previous["queue_panels"]
    assert sum(len(panel["queues"]) for panels in protocol["queue_panels"].values() for panel in panels) == 1143
    assert protocol["budgets"] == {"4": 144, "8": 288, "16": 576}
    assert protocol["reference_budget"] == 4096
    assert protocol["retained_witnesses"] == [
        {"node": "gpu_a", "lanes": 8, "queues": [6, 6, 37]},
        {"node": "gpu_b", "lanes": 16, "queues": [35, 32, 77]},
    ]


def test_tiny_r4_runs_four_independent_selectors_and_checks_fraction_oracle(model, monkeypatch):
    protocol = tiny_protocol(model)
    calls = []
    for name, label in (
        ("select_canonical_count_space_bnb_frame", "reference"),
        ("select_shared_threshold_canonical_requirement_dominance_frame", "old"),
        ("select_fractional_shared_threshold_canonical_requirement_dominance_frame", "fractional"),
        ("select_integer_capacity_shared_threshold_frame", "capacity"),
    ):
        original = getattr(screen, name)

        def tracked(options, queue, lanes, budget, selector=original, kind=label):
            calls.append((kind, tuple(queue), lanes, budget))
            return selector(options, queue, lanes, budget)

        monkeypatch.setattr(screen, name, tracked)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    assert row["queue_count"] == row["oracle_queue_count"] == 2
    assert [(kind, lanes, budget) for kind, _, lanes, budget in calls] == [
        ("reference", 4, 4096), ("old", 4, 144), ("fractional", 4, 144), ("capacity", 4, 144)] * 2
    for label in screen.LABELS:
        assert row[label]["calls"] == row[label]["canonical_exact_count"] == 2
        assert row[label]["failures"] == row[label]["canonical_action_mismatch_count"] == 0
    result = screen.summarize((model,), protocol, [row])
    assert result["development_action_gate_passed"]
    assert result["selector_calls"] == 8


@pytest.mark.parametrize("failure", ("invalid_new", "invalid_control", "incomplete_reference", "unresolved_candidate", "action_mismatch"))
def test_no_false_advance_on_certificate_or_action_failure(model, tiny_result, failure):
    protocol, row = deepcopy(tiny_result)
    if failure == "invalid_new":
        row["integer_capacity"]["failures"] = 1
    elif failure == "invalid_control":
        row["fractional_shared"]["failures"] = 1
    elif failure == "incomplete_reference":
        row["uncertified_reference_count"] = 1
    elif failure == "unresolved_candidate":
        row["integer_capacity"]["canonical_exact_count"] -= 1
    else:
        row["integer_capacity"]["canonical_action_mismatch_count"] = 1
    result = screen.summarize((model,), protocol, [row])
    assert not result["development_action_gate_passed"]
    assert not result["decision"].endswith("ADVANCE")


def test_false_exact_flag_is_checked_against_oracle(model, monkeypatch):
    original = screen.select_integer_capacity_shared_threshold_frame

    def false_exact(options, queue, lanes, budget):
        result = original(options, queue, lanes, budget)
        selected = replace(result.selection, option_indices=(None,) * lanes,
                           option_counts=(0,) * len(options), service=(0, 0, 0),
                           duration=1.0, score=0.0, upper_bound=0.0, gap=0.0,
                           epsilon=0.0, exact=True, score_exact=True, canonical_exact=True)
        return replace(result, selection=selected)

    monkeypatch.setattr(screen, "select_integer_capacity_shared_threshold_frame", false_exact)
    protocol = tiny_protocol(model)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    checks = row["integer_capacity"]["first_failure"]["checks"]
    assert checks["false_score_exact"] and checks["false_canonical_exact"]
    assert screen.summarize((model,), protocol, [row])["decision"] == "INVALID_EXECUTION"


def test_unresolved_controls_do_not_block_exact_improved_candidate(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    for control in screen.CONTROLS:
        row[control]["canonical_exact_count"] = 0
        row[control]["score_mismatch_count"] = 1
        for counter in screen.COMPARISON_COUNTERS:
            row[control]["search_work"][counter] = row["integer_capacity"]["search_work"][counter] + 10
    result = screen.summarize((model,), protocol, [row])
    assert result["decision"] == "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE"


@pytest.mark.parametrize("control", screen.CONTROLS)
def test_each_control_cell_cost_gate_blocks_despite_pooled_savings(model, tiny_result, control):
    protocol, row = tiny_result
    other = replace(model, node="other")
    rows = [deepcopy(row), deepcopy(row)]
    rows[1]["node"] = other.node
    for item, old, new in ((rows[0], 10, 11), (rows[1], 100, 50)):
        for counter in screen.COMPARISON_COUNTERS:
            for label in screen.CONTROLS:
                item[label]["search_work"][counter] = old if label == control else 1000
            item["integer_capacity"]["search_work"][counter] = new
    result = screen.summarize((model, other), protocol, rows)
    assert result["development_action_gate_passed"]
    assert not result["search_cost_gate_by_control"][control]
    assert not result["development_search_cost_gate_passed"]
    assert result["decision"] == "SEARCH_COST_REVIEW_REQUIRED"


def test_incomplete_reference_never_escalates_budget_or_rescues(model, monkeypatch):
    protocol = tiny_protocol(model)
    original = screen.select_canonical_count_space_bnb_frame
    calls = []

    def unresolved(options, queue, lanes, budget):
        calls.append(budget)
        selected = original(options, queue, lanes, budget)
        return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=selected.score + 1, gap=1.0, epsilon=1.0 / sum(queue))

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", unresolved)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    assert calls == [4096, 4096]
    assert row["uncertified_reference_count"] == 2
    assert row["reference"]["failures"] == 0
    assert screen.summarize((model,), protocol, [row])["decision"] == "INCOMPLETE_REFERENCE"


def test_capacity_work_is_separate_and_named_witnesses_are_retained(model, monkeypatch):
    protocol = tiny_protocol(model)
    protocol["retained_witnesses"] = [{"node": model.node, "lanes": 4, "queues": [3, 3, 0]}]
    original = screen.select_integer_capacity_shared_threshold_frame

    def accounted(options, queue, lanes, budget):
        result = original(options, queue, lanes, budget)
        return replace(result, capacity_propagation=replace(result.capacity_propagation,
            region_visits=2, requirement_coordinate_reads=7, tightened_bounds=3, removed_count_values=5))

    monkeypatch.setattr(screen, "select_integer_capacity_shared_threshold_frame", accounted)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    result = screen.summarize((model,), protocol, [row])
    expected = dict(zip(screen.CAPACITY_COUNTERS, (4, 14, 6, 10), strict=True))
    assert result["capacity_propagation"] == row["capacity_propagation"] == expected
    assert not set(screen.CAPACITY_COUNTERS) & set(result["comparison_search_work"]["integer_capacity"])
    witness, = row["panels"][0]["retained_witnesses"]
    assert witness["queues"] == (3, 3, 0)
    assert set(witness) == {"queues", *screen.LABELS}


@pytest.mark.parametrize("change", ("source", "budget", "queue"))
def test_registration_rejects_changed_execution_before_selector_calls(model, tmp_path, monkeypatch, change):
    output, source = tmp_path / "development.json", tmp_path / "selector.py"
    source.write_text("frozen source\n")
    monkeypatch.setattr(screen, "EXECUTION_SOURCES", (str(source),))
    monkeypatch.setattr(screen, "load_catalogues", lambda: (model,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    path = output.with_name(output.stem + "_protocol.json")
    protocol = json.loads(path.read_text())
    if change == "source":
        source.write_text("changed source\n")
    elif change == "budget":
        protocol["budgets"]["4"] += 1
    else:
        protocol["queue_panels"]["4"][0]["queues"][0][0] += 1
    path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("changed protocol reached selector calls")

    monkeypatch.setattr(screen, "screen_cell", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()
