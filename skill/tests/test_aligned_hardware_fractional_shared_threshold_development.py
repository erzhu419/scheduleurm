from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import aligned_hardware_fractional_shared_threshold_development as screen


@pytest.fixture
def model():
    return FrameModel(
        node="fractional_test", scenario_id="small_count_space", lanes=2,
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


def test_roster_is_exact_union_of_old_grid_and_promoted_confirmation(model):
    protocol = screen.build_protocol((model,))
    previous = json.loads(screen.PRIOR_CONFIRMATION.read_text())
    for lanes in screen.LANES:
        panels = protocol["queue_panels"][str(lanes)]
        assert panels[0]["queues"] == list(map(list, screen.scaled_queues(lanes)))
        assert [panel["queues"] for panel in panels[1:]] == [panel["queues"] for panel in previous["queue_panels"][str(lanes)]]
        queues = [tuple(queue) for panel in panels for queue in panel["queues"]]
        assert len(queues) == len(set(queues)) == 381
    assert protocol["budgets"] == {"4": 144, "8": 288, "16": 576}
    assert protocol["reference_budget"] == 4096


def test_tiny_r4_runs_three_independent_selectors_and_checks_fraction_oracle(model, monkeypatch):
    protocol = tiny_protocol(model)
    calls = []
    for name, label in (
        ("select_canonical_count_space_bnb_frame", "reference"),
        ("select_shared_threshold_canonical_requirement_dominance_frame", "old"),
        ("select_fractional_shared_threshold_canonical_requirement_dominance_frame", "fractional"),
    ):
        original = getattr(screen, name)

        def tracked(options, queue, lanes, budget, selector=original, kind=label):
            calls.append((kind, tuple(queue), lanes, budget))
            return selector(options, queue, lanes, budget)

        monkeypatch.setattr(screen, name, tracked)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    assert row["queue_count"] == row["oracle_queue_count"] == 2
    assert [(kind, lanes, budget) for kind, _, lanes, budget in calls] == [
        ("reference", 4, 4096), ("old", 4, 144), ("fractional", 4, 144)] * 2
    for label in screen.LABELS:
        assert row[label]["calls"] == row[label]["canonical_exact_count"] == 2
        assert row[label]["failures"] == row[label]["canonical_action_mismatch_count"] == 0
    assert screen.summarize((model,), protocol, [row])["development_action_gate_passed"]


@pytest.mark.parametrize("failure", ("invalid_certificate", "incomplete_reference", "unresolved_candidate", "action_mismatch"))
def test_no_false_advance_on_certificate_or_action_failure(model, tiny_result, failure):
    protocol, row = deepcopy(tiny_result)
    if failure == "invalid_certificate":
        row["fractional_shared"]["failures"] = 1
    elif failure == "incomplete_reference":
        row["uncertified_reference_count"] = 1
    elif failure == "unresolved_candidate":
        row["fractional_shared"]["canonical_exact_count"] -= 1
    else:
        row["fractional_shared"]["canonical_action_mismatch_count"] = 1
    result = screen.summarize((model,), protocol, [row])
    assert not result["development_action_gate_passed"]
    assert not result["decision"].endswith("ADVANCE")


def test_old_unresolved_control_does_not_block_exact_improved_candidate(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    row["old_shared"]["canonical_exact_count"] = 0
    row["old_shared"]["score_mismatch_count"] = 1
    for counter in screen.COMPARISON_COUNTERS:
        row["old_shared"]["search_work"][counter] = row["fractional_shared"]["search_work"][counter] + 10
    result = screen.summarize((model,), protocol, [row])
    assert result["decision"] == "FRACTIONAL_BRANCH_DEVELOPMENT_ADVANCE"


def test_cell_cost_regression_blocks_advance_despite_pooled_savings(model, tiny_result):
    protocol, row = tiny_result
    other = replace(model, node="other")
    rows = [deepcopy(row), deepcopy(row)]
    rows[1]["node"] = other.node
    for item, old, new in ((rows[0], 10, 11), (rows[1], 100, 50)):
        for counter in screen.COMPARISON_COUNTERS:
            item["old_shared"]["search_work"][counter] = old
            item["fractional_shared"]["search_work"][counter] = new
    result = screen.summarize((model, other), protocol, rows)
    assert result["development_action_gate_passed"]
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
