from copy import deepcopy
from dataclasses import replace
from itertools import product
import json

import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import aligned_hardware_shared_threshold_queue_confirmation as screen
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import scaled_queues


@pytest.fixture
def model():
    return FrameModel(
        node="confirmation_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(
            FrameOption("a_early", (1, 0, 0), 3.0, ((3.0,), (), ())),
            FrameOption("a_late", (1, 0, 0), 1.0, ((1.0,), (), ())),
            FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ())),
        ),
    )


@pytest.mark.parametrize("lanes", (4, 8, 16))
def test_sampler_replays_distinct_new_integer_queues(lanes):
    queues = screen.sample_queues(lanes, 123, count=64)
    assert queues == screen.sample_queues(lanes, 123, count=64)
    assert len(queues) == len(set(map(tuple, queues))) == 64
    old = set(scaled_queues(lanes)) | set(product((0, 1, 3, 4, 10), repeat=3))
    assert not set(map(tuple, queues)) & old
    assert all(len(queue) == 3 and all(isinstance(value, int) and 0 <= value <= 5 * lanes for value in queue)
               for queue in queues)
    following = screen.sample_queues(lanes, 124, count=64, excluded=queues)
    assert not set(map(tuple, queues)) & set(map(tuple, following))


def test_registered_rosters_are_shared_across_models_and_disjoint_across_panels(model):
    single = screen.build_protocol((model,))
    two = screen.build_protocol((model, replace(model, node="other_hardware")))
    assert single["queue_panels"] == two["queue_panels"]
    assert json.loads(json.dumps(two)) == two
    for lanes_text, panels in single["queue_panels"].items():
        lanes = int(lanes_text)
        assert len(panels) == 4
        assert len({panel["seed"] for panel in panels}) == 4
        previous = set()
        for panel in panels:
            queues = set(map(tuple, panel["queues"]))
            assert len(queues) == 64
            assert not queues & previous
            assert tuple(map(tuple, panel["queues"])) == tuple(map(tuple, screen.sample_queues(
                lanes, panel["seed"], count=64, excluded=previous)))
            previous.update(queues)


@pytest.mark.parametrize("field", ("roster", "reference_budget", "deployment_budget", "execution_source"))
def test_cli_rejects_registered_roster_or_budget_drift_before_search(model, tmp_path, monkeypatch, field):
    output = tmp_path / "confirmation.json"
    source = tmp_path / "test_selector_source.py"
    source.write_text("original frozen selector\n")
    monkeypatch.setattr(screen, "EXECUTION_SOURCES", (str(source),))
    monkeypatch.setattr(screen, "load_catalogues", lambda: (model,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    protocol_path = output.with_name(output.stem + "_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if field == "roster":
        protocol["queue_panels"]["4"][0]["queues"][0][0] += 1
    elif field == "deployment_budget":
        protocol["budgets"]["4"] += 1
    elif field == "execution_source":
        source.write_text("changed selector after registration\n")
    else:
        protocol[field] += 1
    protocol_path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("changed frozen protocol reached search")

    monkeypatch.setattr(screen, "screen_cell", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()


def _tiny_protocol(model):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[4], registered_at_utc="test")
    protocol["queue_panels"] = {"4": [
        {"seed": 101, "queues": [[3, 3, 0], [2, 4, 1]]},
        {"seed": 102, "queues": [[4, 3, 2], [0, 3, 2]]},
    ]}
    return protocol


@pytest.fixture
def tiny_result(model):
    protocol = _tiny_protocol(model)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    return protocol, row


def test_tiny_r4_cell_checks_actual_original_actions_against_oracle(model, monkeypatch):
    protocol = _tiny_protocol(model)
    full_selector = screen.select_canonical_count_space_bnb_frame
    shared_selector = screen.select_shared_threshold_canonical_requirement_dominance_frame
    calls = []

    def full(options, queues, lanes, budget):
        calls.append(("full", tuple(queues), lanes, budget))
        assert options == model.options
        return full_selector(options, queues, lanes, budget)

    def shared(options, queues, lanes, budget):
        calls.append(("shared", tuple(queues), lanes, budget))
        assert options == model.options
        return shared_selector(options, queues, lanes, budget)

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", full)
    monkeypatch.setattr(screen, "select_shared_threshold_canonical_requirement_dominance_frame", shared)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    assert row["queue_count"] == row["oracle_queue_count"] == 4
    assert len(calls) == 12
    for panel in protocol["queue_panels"]["4"]:
        for queue in panel["queues"]:
            assert [(label, lanes, budget) for label, called_queue, lanes, budget in calls if called_queue == tuple(queue)] == [
                ("full", 4, 4096), ("full", 4, 144), ("shared", 4, 144)]
    for label in ("reference", "full", "shared"):
        assert row[label]["calls"] == row[label]["canonical_exact_count"] == 4
        assert row[label]["failures"] == row[label]["canonical_action_mismatch_count"] == 0
    summary = screen.summarize((model,), protocol, [row])
    assert summary["certificate_checks_passed"]
    assert summary["references_complete"]
    assert summary["deployment_action_gate_passed"]


@pytest.mark.parametrize("failure", ("unresolved_reference", "uncertified_canonical"))
def test_confirmation_cannot_advance_on_incomplete_certificates(model, tiny_result, failure):
    protocol, original = tiny_result
    row = deepcopy(original)
    if failure == "unresolved_reference":
        row["uncertified_reference_count"] += 1
        row["panels"][0]["uncertified_reference_count"] += 1
    else:
        row["shared"]["canonical_exact_count"] -= 1
        row["panels"][0]["shared"]["canonical_exact_count"] -= 1
    summary = screen.summarize((model,), protocol, [row])
    assert not summary["deployment_action_gate_passed"]
    assert not summary["decision"].endswith("ADVANCE")


def test_one_node_cost_regression_blocks_advance_despite_pooled_savings(model, tiny_result):
    protocol, original = tiny_result
    other = replace(model, node="other_hardware")
    rows = [deepcopy(original), deepcopy(original)]
    rows[1]["node"] = other.node
    for row, full, shared in ((rows[0], 10, 11), (rows[1], 100, 50)):
        for counter in ("lp_calls", "lp_matrix_coefficients"):
            row["full"]["search_work"][counter] = full
            row["shared"]["search_work"][counter] = shared
            for index, panel in enumerate(row["panels"]):
                panel["full"]["search_work"][counter] = full if index == 0 else 0
                panel["shared"]["search_work"][counter] = shared if index == 0 else 0
    summary = screen.summarize((model, other), protocol, rows)
    assert not summary["deployment_search_cost_gate_passed"]
    assert not summary["decision"].endswith("ADVANCE")


def test_seed_results_are_descriptive_while_node_lane_aggregate_controls_gate(model, tiny_result):
    protocol, original = tiny_result
    row = deepcopy(original)
    for counter in ("lp_calls", "lp_matrix_coefficients"):
        row["panels"][0]["full"]["search_work"][counter] = 10
        row["panels"][0]["shared"]["search_work"][counter] = 11
        row["panels"][1]["full"]["search_work"][counter] = 100
        row["panels"][1]["shared"]["search_work"][counter] = 50
        row["full"]["search_work"][counter] = 110
        row["shared"]["search_work"][counter] = 61
    summary = screen.summarize((model,), protocol, [row])
    assert summary["deployment_search_cost_gate_passed"]
    assert summary["deployment_action_gate_passed"]


def test_uncertified_reference_never_triggers_larger_budget_or_rescue(model, monkeypatch):
    protocol = _tiny_protocol(model)
    selector = screen.select_canonical_count_space_bnb_frame
    calls = []

    def incomplete_reference(options, queues, lanes, budget):
        calls.append(budget)
        selected = selector(options, queues, lanes, budget)
        if budget != 4096:
            return selected
        return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=selected.score + 1, gap=1.0,
                       epsilon=1.0 / sum(queues))

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", incomplete_reference)
    row = screen.screen_cell(model, 4, protocol, progress=False)
    assert calls == [4096, 144] * 4
    assert row["uncertified_reference_count"] == 4
    assert row["reference"]["failures"] == 0
    summary = screen.summarize((model,), protocol, [row])
    assert not summary["references_complete"]
    assert not summary["decision"].endswith("ADVANCE")
