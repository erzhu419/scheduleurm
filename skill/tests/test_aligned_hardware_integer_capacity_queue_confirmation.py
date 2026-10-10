from copy import deepcopy
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_queue_confirmation as screen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@pytest.fixture
def model():
    return FrameModel(
        node="confirmation_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (1, 0, 0), 1.0, ((1.0,), (), ())),),
    )


def test_new_roster_is_deterministic_disjoint_and_keeps_frozen_budgets(model):
    protocol = screen.build_protocol((model,))
    assert protocol == screen.build_protocol((model,))
    previous = json.loads(screen.DEVELOPMENT_PROTOCOL.read_text())
    assert protocol["panel_seeds"] == [2026092705, 2026092706, 2026092707, 2026092708]
    assert protocol["budgets"] == previous["budgets"] == {"4": 144, "8": 288, "16": 576}
    assert protocol["reference_budget"] == previous["reference_budget"] == 4096
    for lanes in (4, 8, 16):
        seen = {tuple(queue) for panel in previous["queue_panels"][str(lanes)] for queue in panel["queues"]}
        fresh = []
        for panel in protocol["queue_panels"][str(lanes)]:
            queues = list(map(tuple, panel["queues"]))
            assert queues == list(screen.sample_queues(lanes, panel["seed"], 64, excluded=seen))
            assert not set(queues) & seen
            assert all(0 <= value <= 5 * lanes for queue in queues for value in queue)
            seen.update(queues)
            fresh.extend(queues)
        assert len(fresh) == len(set(fresh)) == 256
    assert protocol["retained_witnesses"] == []


@pytest.mark.parametrize("failure", ("missing", "decision", "certificate_checks_passed", "references_complete", "development_action_gate_passed", "development_search_cost_gate_passed"))
def test_registration_waits_for_actual_development_pass(tmp_path, monkeypatch, failure):
    result_path, output = tmp_path / "development.json", tmp_path / "confirmation.json"
    result = {"decision": "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE", "certificate_checks_passed": True,
              "references_complete": True, "development_action_gate_passed": True,
              "development_search_cost_gate_passed": True}
    if failure != "missing":
        result[failure] = "SEARCH_COST_REVIEW_REQUIRED" if failure == "decision" else False
        result_path.write_text(json.dumps(result))
    monkeypatch.setattr(screen, "DEVELOPMENT_OUTPUT", result_path)
    monkeypatch.setattr("sys.argv", ["confirmation", "--output", str(output), "--register"])

    def unexpected_build(*args):
        raise AssertionError("failed development reached confirmation registration")

    monkeypatch.setattr(screen, "build_protocol", unexpected_build)
    with pytest.raises(ValueError, match="development"):
        screen.main()
    assert not output.with_name(output.stem + "_protocol.json").exists()


def test_complete_development_pass_is_accepted(tmp_path, monkeypatch):
    path = tmp_path / "development.json"
    path.write_text(json.dumps({"decision": "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE",
        "certificate_checks_passed": True, "references_complete": True,
        "development_action_gate_passed": True, "development_search_cost_gate_passed": True}))
    monkeypatch.setattr(screen, "DEVELOPMENT_OUTPUT", path)
    screen.require_development_advance()


@pytest.fixture
def tiny_result(model):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[4], registered_at_utc="test")
    protocol["queue_panels"] = {"4": [{"panel": "fresh_test", "seed": 1, "queues": [[2, 0, 0]]}]}
    row = screen.screen_cell(model, 4, protocol, progress=False)
    for control in screen.development.CONTROLS:
        for key in screen.development.COMPARISON_COUNTERS:
            row[control]["search_work"][key] = row["integer_capacity"]["search_work"][key] + 10
    return protocol, row


@pytest.mark.parametrize("failure", (None, "old_shared", "fractional_shared", "action", "invalid", "reference"))
def test_confirmation_keeps_both_cost_gates_and_exactness_gates(model, tiny_result, failure):
    protocol, row = deepcopy(tiny_result)
    if failure in screen.development.CONTROLS:
        row[failure]["search_work"]["lp_calls"] = 0
    elif failure == "action":
        row["integer_capacity"]["canonical_exact_count"] = 0
    elif failure == "invalid":
        row["integer_capacity"]["failures"] = 1
    elif failure == "reference":
        row["uncertified_reference_count"] = 1
    result = screen.summarize((model,), protocol, [row])
    assert "development_action_gate_passed" not in result
    assert "development_search_cost_gate_passed" not in result
    if failure is None:
        assert result["decision"] == "INTEGER_CAPACITY_FRESH_QUEUE_CONFIRMATION_ADVANCE"
        assert result["confirmation_action_gate_passed"] and result["confirmation_search_cost_gate_passed"]
    else:
        assert not result["decision"].endswith("ADVANCE")
        if failure in screen.development.CONTROLS:
            assert result["decision"] == "SEARCH_COST_REVIEW_REQUIRED"
        elif failure == "action":
            assert result["decision"] == "CONFIRMATION_ACTION_REVIEW_REQUIRED"
