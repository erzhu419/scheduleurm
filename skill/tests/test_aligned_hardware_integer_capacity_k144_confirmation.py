from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_k144_confirmation as screen
from algorithm.experiments import aligned_hardware_integer_capacity_budget_frontier as frontier
from algorithm.experiments import aligned_hardware_integer_capacity_large_r as large_r
from algorithm.experiments import integer_capacity_k144_confirmation_summary as summary
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@pytest.fixture
def model(tmp_path, monkeypatch):
    model = FrameModel(
        node="confirmation_test", scenario_id="small_count_space", lanes=2,
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
        "queue_panels": frontier.queue_design(large_r.queue_design()[0]),
        "retained_witnesses": frontier.RETAINED_WITNESSES,
    }))
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({
        "decision": "INTEGER_CAPACITY_BUDGET_FRONTIER_CHARACTERIZED", "queue_count": 1236,
        "certificate_checks_passed": True, "references_complete": True,
        "high_cap_action_gate_passed": True,
        "curves": [{"budget": 144, "queue_count": 1236, "label_rates": {"integer_capacity": {
            "certified_canonical_action_equal_count": 1236,
            "observed_canonical_action_equal_count": 1236,
        }}}],
    }))
    static_paths = []
    for index, queue in enumerate(((1, 2, 3), (4, 5, 6))):
        path = tmp_path / f"static_{index}.json"
        path.write_text(json.dumps({"queue_panels": {
            str(lanes): [{"queues": [queue]}] for lanes in screen.LANES
        }}))
        static_paths.append(path)
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    monkeypatch.setattr(screen, "PRIOR_PROTOCOL", previous)
    monkeypatch.setattr(screen, "PRIOR_OUTPUT", prior_output)
    monkeypatch.setattr(screen, "STATIC_PROTOCOLS", tuple(static_paths))
    monkeypatch.setattr(screen, "EXTRA_SOURCES", ())
    monkeypatch.setattr(large_r, "runtime_provenance", lambda: {})
    monkeypatch.setattr(screen, "runtime_setup", lambda: runtime)
    return model


def tiny_protocol(model):
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[8], registered_at_utc="test", retained_witnesses=[])
    protocol["queue_panels"] = {"8": [
        {"panel": "fresh_test", "cohort": "fresh", "states": [
            {"state_id": "fresh:0", "queue": [3, 3, 0], "cohort": "fresh"},
        ]},
        {"panel": "hard_test", "cohort": "regression", "states": [
            {"state_id": "hard:0", "queue": [4, 3, 0], "cohort": "regression"},
        ]},
    ]}
    return protocol


@pytest.fixture
def tiny_result(model):
    protocol = tiny_protocol(model)
    return protocol, screen.screen_cell(model, 8, protocol, progress=False)


def record(row, state=0, label="integer_capacity"):
    return row["states"][state]["budgets"]["144"][label]


def make_unresolved(item):
    item["selection"].update(exact=False, score_exact=True, canonical_exact=False, gap=0.0)
    item["metrics"].update(score_exact_count=1, canonical_exact_count=0, gap_sum=0.0, max_gap=0.0)


def test_fresh_roster_is_deterministic_disjoint_and_retains_hard_cases(model):
    protocol = screen.build_protocol((model,))
    assert protocol == screen.build_protocol((model,))
    assert protocol["lanes"] == [8, 16, 32, 64]
    assert protocol["budget_grid"] == [144]
    assert protocol["reference_budget"] == 4096
    assert all(caps == [144] for caps in protocol["unique_budgets"].values())
    previous = json.loads(screen.PRIOR_PROTOCOL.read_text())
    excluded = screen.static_exclusions(previous)
    fresh_total = regression_total = 0
    for lanes in screen.LANES:
        panels = protocol["queue_panels"][str(lanes)]
        fresh = [state for panel in panels if panel["cohort"] == "fresh" for state in panel["states"]]
        regression = [state for panel in panels if panel["cohort"] == "regression" for state in panel["states"]]
        queues = {tuple(state["queue"]) for state in fresh}
        assert len(fresh) == len(queues) == 256
        assert queues.isdisjoint(excluded[str(lanes)])
        assert all(state["cohort"] == "fresh" for state in fresh)
        assert all(state["cohort"] == "regression" for state in regression)
        expected = [state for panel in previous["queue_panels"][str(lanes)]
                    if panel["panel"].startswith("hard_") for state in panel["states"]]
        assert [{key: value for key, value in state.items() if key != "cohort"}
                for state in regression] == expected
        assert len({state["state_id"] for state in fresh + regression}) == len(fresh + regression)
        fresh_total += len(fresh)
        regression_total += len(regression)
    assert fresh_total * 4 == 4096
    assert regression_total * 4 == 324


def test_one_reference_and_three_cold_calls_per_state(model, monkeypatch):
    protocol, calls = tiny_protocol(model), []
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
    assert calls == [(label, queue, 8, 4096 if label == "reference" else 144)
                     for queue in ((3, 3, 0), (4, 3, 0))
                     for label in ("reference", "old_shared", "fractional_shared", "integer_capacity")]
    result = summary.summarize((model,), protocol, [row])
    assert result["executed_reference_calls"] == 2
    assert result["executed_candidate_calls"] == 6
    assert result["executed_selector_calls"] == 8
    assert result["fresh"]["queue_count"] == result["regression"]["queue_count"] == 1
    assert result["decision"] == "INTEGER_CAPACITY_K144_CONFIRMED"


@pytest.mark.parametrize("state,decision", [(0, "K144_FRESH_CONFIRMATION_HOLD"), (1, "K144_REGRESSION_HOLD")])
def test_action_agreement_does_not_rescue_unclosed_cohort(model, tiny_result, state, decision):
    protocol, row = deepcopy(tiny_result)
    make_unresolved(record(row, state))
    row["states"][state]["cohort"] = "regression" if state == 0 else "fresh"
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == decision
    assert result["certificate_checks_passed"] and result["references_complete"]
    cohort = result["fresh" if state == 0 else "regression"]
    assert not cohort["action_gate_passed"]
    rates = cohort["curves"][0]["label_rates"]["integer_capacity"]
    assert rates["observed_canonical_action_equal_count"] == 1
    assert rates["certified_canonical_action_equal_count"] == 0
    assert cohort["curves"][0]["normalized_gaps"]["integer_capacity"]["zero_gap_uncertified_count"] == 1


def test_honestly_unclosed_controls_do_not_invalidate_candidate(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    for state in (0, 1):
        for label in ("old_shared", "fractional_shared"):
            make_unresolved(record(row, state, label))
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INTEGER_CAPACITY_K144_CONFIRMED"
    assert result["confirmation_action_gate_passed"] and result["regression_action_gate_passed"]


def test_cost_regression_is_secondary_to_fixed_budget_closure(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    candidate = record(row)["metrics"]["search_work"]
    for counter in runtime.COMPARISON_COUNTERS:
        candidate[counter] = 1 + max(record(row, label=label)["metrics"]["search_work"][counter]
                                     for label in runtime.CONTROLS)
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INTEGER_CAPACITY_K144_CONFIRMED"
    assert not result["confirmation_search_cost_gate_passed"]
    assert all(result["fresh"]["cost_regressions"][control][counter]["count"] == 1
               for control in runtime.CONTROLS for counter in runtime.COMPARISON_COUNTERS)


def test_incomplete_reference_cannot_be_rescued_by_candidate(model, monkeypatch):
    original, budgets = runtime.select_canonical_count_space_bnb_frame, []

    def unresolved(options, queue, lanes, budget):
        budgets.append(budget)
        selected = original(options, queue, lanes, budget)
        return replace(selected, exact=False, score_exact=False, canonical_exact=False,
                       upper_bound=selected.score + 1, gap=1.0, epsilon=1.0 / sum(queue))

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", unresolved)
    protocol = tiny_protocol(model)
    row = screen.screen_cell(model, 8, protocol, progress=False)
    assert budgets == [4096, 4096]
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert not result["confirmation_action_gate_passed"]
    assert not result["regression_action_gate_passed"]
    assert result["fresh"]["curves"][0]["label_rates"]["integer_capacity"]["canonical_exact_count"] == 1


@pytest.mark.parametrize("change", ["missing_state", "duplicate_cell", "call_count", "budget", "cohort", "no_regression"])
def test_lost_coverage_or_changed_cohort_is_rejected(model, tiny_result, change):
    protocol, row = deepcopy(tiny_result)
    rows = [row]
    if change == "missing_state":
        row["states"].pop()
    elif change == "duplicate_cell":
        rows.append(deepcopy(row))
    elif change == "call_count":
        record(row)["metrics"]["calls"] = 2
    elif change == "budget":
        protocol["budget_grid"] = [288]
    elif change == "cohort":
        protocol["queue_panels"]["8"][0]["cohort"] = "pooled"
    else:
        protocol["queue_panels"]["8"][1]["cohort"] = "fresh"
    with pytest.raises(ValueError):
        summary.summarize((model,), protocol, rows)


def test_invalid_control_call_cannot_be_hidden_by_candidate_success(model, tiny_result):
    protocol, row = deepcopy(tiny_result)
    record(row, label="old_shared")["metrics"]["failures"] = 1
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert not result["certificate_checks_passed"]
    assert not result["confirmation_action_gate_passed"]


@pytest.mark.parametrize("change", ["source", "model", "queue", "budget", "cohort"])
def test_protocol_changes_stop_before_execution(model, tmp_path, monkeypatch, change):
    output = tmp_path / "confirmation.json"
    monkeypatch.setattr(runtime, "load_catalogues", lambda: (model,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    path = output.with_name(output.stem + "_protocol.json")
    protocol = json.loads(path.read_text())
    if change == "source":
        (tmp_path / "selector.py").write_text("changed source\n")
    elif change == "model":
        monkeypatch.setattr(runtime, "load_catalogues", lambda: (replace(model, lanes=3),))
    elif change == "queue":
        protocol["queue_panels"]["8"][0]["states"][0]["queue"][0] += 1
    elif change == "budget":
        protocol["budget_grid"] = [288]
    else:
        protocol["queue_panels"]["8"][0]["cohort"] = "regression"
    path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("changed protocol reached selector calls")

    monkeypatch.setattr(screen, "screen_cell", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--run"])
    with pytest.raises(ValueError):
        screen.main()
    assert not output.exists()
