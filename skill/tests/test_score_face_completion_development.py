from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import gzip
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.score_face_completion_development_summary import summarize
import algorithm.score_face_canonical_completion_frame_candidates as completion


@pytest.fixture
def case():
    model = FrameModel(
        node="completion_test", scenario_id="synthetic", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a_slow", (1, 0, 0), 3.0, ((3.0,), (), ())),
                 FrameOption("a_fast", (1, 0, 0), 1.0, ((1.0,), (), ())),
                 FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ()))),
    )
    queue = (3, 3, 0)
    reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, 2, 4096)
    baseline = runtime.select_integer_capacity_shared_threshold_frame(model.options, queue, 2, 144)
    witness = {"name": "synthetic", "node": model.node, "lanes": 2, "queue": list(queue), "state_id": "s:0"}
    score = Fraction(sum(q * s for q, s in zip(queue, reference.service))) / Fraction.from_float(reference.duration)
    protocol = {
        "experiment": "synthetic_completion", "registered_at_utc": "new", "lanes": [2],
        "budget": 144, "reference_budget": 4096, "canonical_state_budget": 100000,
        "oracle_state_budget": 5000000, "oracle_witnesses": [witness],
        "prior_decision": "QUEUE_AWARE_CANONICAL_ACTION_HOLD", "limitations": "synthetic test",
        "queue_panels": {"2": [{"panel": "tiny", "cohort": "reused_confirmation", "states": [
            {"state_id": "s:0", "queue": list(queue), "cohort": "reused_confirmation"}]}]},
    }
    oracles = [{**witness, "result": {
        "complete": True, "canonical_exact": True, "state_visit_budget": 5000000,
        "score_fraction": [score.numerator, score.denominator],
        "duration_fraction": [Fraction.from_float(reference.duration).numerator, Fraction.from_float(reference.duration).denominator],
        "option_indices": list(reference.option_indices),
    }}]
    return model, protocol, baseline, reference, oracles


def rows_for(case, monkeypatch, *, open_baseline=False, state_budget=None, missing_reference=False):
    model, protocol, baseline, reference, oracles = deepcopy(case)
    queue = tuple(protocol["queue_panels"]["2"][0]["states"][0]["queue"])
    if open_baseline:
        baseline = replace(baseline, selection=replace(baseline.selection, exact=False,
            canonical_exact=False, canonical_tie_regions_remaining=2,
            frontier_regions_remaining=2, inherited_unsolved_regions=1))
    if state_budget is not None:
        protocol["canonical_state_budget"] = state_budget
    monkeypatch.setattr(completion, "select_integer_capacity_shared_threshold_frame", lambda *args: baseline)
    result = completion.select_score_face_canonical_completion_frame(
        model.options, queue, 2, protocol["budget"], protocol["canonical_state_budget"])
    expected = runtime.reference_key(reference, queue, len(model.options))
    if missing_reference:
        reference = replace(reference, exact=False, canonical_exact=False)
        if not oracles or not oracles[0]["result"]["canonical_exact"]:
            expected = None
    def metrics(selected, preprocessing=None):
        value = runtime._metrics()
        runtime._record(value, selected, model, queue, 2, selected.lp_node_budget,
                        expected, preprocessing, reference)
        return value
    prep, capacity = asdict(baseline.preprocessing), asdict(baseline.capacity_propagation)
    state = {
        "state_id": "s:0", "panel": "tiny", "cohort": "reused_confirmation", "queue": list(queue),
        "reference": asdict(reference), "reference_metrics": metrics(reference),
        "reference_complete": reference.canonical_exact,
        "effective_reference_complete": expected is not None,
        "effective_reference_source": "full_K4096" if reference.canonical_exact else "independent_score_face_oracle" if expected is not None else None,
        "oracle_crosscheck_failed": False,
        "legacy": {"selection": asdict(baseline.selection), "metrics": metrics(baseline.selection, prep), "capacity_propagation": capacity},
        "completed": {"selection": asdict(result.selection), "baseline_selection": asdict(result.baseline_selection),
            "metrics": metrics(result.selection, prep), "capacity_propagation": capacity, "completion": asdict(result.completion)},
        "baseline_metrics": metrics(result.baseline_selection, prep), "baseline_reproduced": True,
    }
    return (model,), protocol, [{"node": model.node, "lanes": 2, "states": [state]}], oracles


def test_skip_preserves_frozen_stage_one_and_separates_reused_calls(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch))
    assert result["decision"] == "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE"
    assert result["executed_selector_calls"] == result["queue_count"] == 1
    assert result["reused_legacy_calls"] == result["reused_reference_calls"] == result["reused_oracle_calls"] == 1
    assert result["executed_oracle_calls"] == 0
    assert result["completion"]["attempted_count"] == 0
    assert result["completion"]["skipped_canonical_complete_count"] == 1
    assert not any(result["completion"]["work"].values())


def test_completion_charges_integer_work_without_adding_a_selector_call(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, open_baseline=True))
    assert result["decision"] == "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE"
    assert result["executed_selector_calls"] == 1
    assert result["completion"]["attempted_count"] == result["completion"]["completed_count"] == 1
    assert result["completion"]["work"]["states_visited"] > 0
    assert result["metrics"]["legacy"]["search_work"] == result["metrics"]["completed"]["search_work"]


def test_honest_state_budget_exhaustion_holds_without_invalidating_execution(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, open_baseline=True, state_budget=0))
    assert result["decision"] == "SCORE_FACE_COMPLETION_HOLD"
    assert result["certificate_checks_passed"]
    assert result["completion"]["exhausted_count"] == 1
    assert result["completion"]["work"]["states_visited"] == 0


def test_independent_reference_supplements_without_rewriting_original_nonclosure(case, monkeypatch):
    result = summarize(*rows_for(case, monkeypatch, missing_reference=True))
    assert result["decision"] == "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE"
    assert not result["original_references_complete"]
    assert result["effective_references_complete"]


def test_candidate_certificate_cannot_supply_its_own_reference(case, monkeypatch):
    case[1]["oracle_witnesses"].clear()
    case[4].clear()
    args = rows_for(case, monkeypatch, missing_reference=True)
    result = summarize(*args)
    assert result["decision"] == "INCOMPLETE_REFERENCE"
    assert not result["effective_references_complete"]
    args[2][0]["states"][0]["effective_reference_complete"] = True
    with pytest.raises(ValueError, match="independent reference"):
        summarize(*args)


@pytest.mark.parametrize("mutation", ("missing_cell", "missing_state", "duplicate_state", "changed_queue", "changed_cohort", "missing_oracle"))
def test_summary_rejects_incomplete_or_changed_coverage(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch)
    rows = args[2]
    if mutation == "missing_cell":
        rows.clear()
    elif mutation == "missing_state":
        rows[0]["states"].clear()
    elif mutation == "duplicate_state":
        rows[0]["states"].append(deepcopy(rows[0]["states"][0]))
    elif mutation == "changed_queue":
        rows[0]["states"][0]["queue"] = [4, 3, 0]
    elif mutation == "changed_cohort":
        rows[0]["states"][0]["cohort"] = "fresh"
    else:
        args[3].clear()
    with pytest.raises(ValueError):
        summarize(*args)


@pytest.mark.parametrize("mutation", ("lp_budget", "state_budget", "oracle_budget", "call_count"))
def test_summary_rejects_changed_budgets_or_duplicate_execution(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch)
    state = args[2][0]["states"][0]
    if mutation == "lp_budget":
        state["completed"]["selection"]["lp_node_budget"] += 1
    elif mutation == "state_budget":
        state["completed"]["completion"]["state_budget"] += 1
    elif mutation == "oracle_budget":
        args[3][0]["result"]["state_visit_budget"] += 1
    else:
        state["completed"]["metrics"]["calls"] = 2
    with pytest.raises(ValueError):
        summarize(*args)


@pytest.mark.parametrize("mutation", ("baseline_flag", "baseline_record", "counter_overrun", "wrong_trigger", "skip_work", "changed_lp_work", "score_target", "live_frontier", "false_change_flag"))
def test_accounting_or_stage_one_drift_invalidates_execution(case, monkeypatch, mutation):
    args = rows_for(case, monkeypatch, open_baseline=mutation in ("score_target", "live_frontier"))
    state = args[2][0]["states"][0]
    work = state["completed"]["completion"]
    if mutation == "baseline_flag":
        state["baseline_reproduced"] = False
    elif mutation == "baseline_record":
        state["legacy"]["selection"]["lp_calls"] += 1
    elif mutation == "counter_overrun":
        work["states_visited"] = work["state_budget"] + 1
    elif mutation == "wrong_trigger":
        work["attempted"] = not work["attempted"]
    elif mutation == "skip_work":
        work["count_branches"] = 1
    elif mutation == "changed_lp_work":
        state["completed"]["selection"]["lp_calls"] += 1
    elif mutation == "score_target":
        work["score_fraction"] = [1, 1]
    elif mutation == "live_frontier":
        state["completed"]["selection"]["frontier_regions_remaining"] = 1
    else:
        work["action_changed"] = True
    result = summarize(*args)
    assert result["decision"] == "INVALID_EXECUTION"
    assert result["completion_accounting_failures"]


def test_reference_contradiction_invalidates_execution(case, monkeypatch):
    args = rows_for(case, monkeypatch)
    args[2][0]["states"][0]["oracle_crosscheck_failed"] = True
    assert summarize(*args)["decision"] == "INVALID_EXECUTION"


def test_runner_executes_one_cold_wrapper_and_reuses_all_comparison_records(case, monkeypatch):
    from algorithm.experiments import aligned_hardware_score_face_completion_development as screen
    from algorithm.experiments import integer_capacity_canonical_oracle as oracle
    models, protocol, previous, oracles = rows_for(case, monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("a historical reference was rerun")
    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", forbidden)
    monkeypatch.setattr(oracle, "canonical_at_certified_score", forbidden)
    calls = []
    def observed(options, queue, lanes, lp_budget, state_budget):
        calls.append((queue, lanes, lp_budget, state_budget))
        return completion.select_score_face_canonical_completion_frame(options, queue, lanes, lp_budget, state_budget)
    row = screen.screen_cell(models[0], 2, protocol, previous[0], oracles, runtime,
                             selector=observed, progress=False)
    assert calls == [((3, 3, 0), 2, 144, 100000)]
    assert row["states"][0]["baseline_reproduced"]
    assert summarize(models, protocol, [row], oracles)["decision"] == "SCORE_FACE_COMPLETION_DEVELOPMENT_ADVANCE"


def test_runner_supplements_incomplete_frozen_reference_and_detects_contradiction(case, monkeypatch):
    from algorithm.experiments import aligned_hardware_score_face_completion_development as screen
    models, protocol, previous, oracles = rows_for(case, monkeypatch, missing_reference=True)
    monkeypatch.setattr(screen, "runtime_provenance", lambda: {})
    row = screen.screen_cell(models[0], 2, protocol, previous[0], oracles, runtime, progress=False)
    assert not row["states"][0]["reference_complete"]
    assert row["states"][0]["effective_reference_source"] == "independent_score_face_oracle"
    assert summarize(models, protocol, [row], oracles)["development_action_gate_passed"]
    models, protocol, previous, oracles = rows_for(case, monkeypatch)
    oracles[0]["result"]["score_fraction"][0] += 1
    row = screen.screen_cell(models[0], 2, protocol, previous[0], oracles, runtime, progress=False)
    assert row["states"][0]["oracle_crosscheck_failed"]
    assert summarize(models, protocol, [row], oracles)["decision"] == "INVALID_EXECUTION"


@pytest.fixture
def saved_case(case, monkeypatch, tmp_path):
    from algorithm.experiments import aligned_hardware_score_face_completion_development as screen
    models, protocol, rows, oracles = rows_for(case, monkeypatch)
    old_protocol = {**protocol, "models": [models[0].snapshot()], "registered_at_utc": "old",
        "execution_sources": {"legacy.py": "frozen source\n"},
        "search_counters": runtime.COUNTERS, "capacity_counters": runtime.CAPACITY_COUNTERS,
        "execution_order": [[models[0].node, 2]]}
    prior_protocol, prior_output, prior_oracles = (tmp_path / name for name in ("prior_protocol.json", "prior.json", "prior_oracles.json"))
    (tmp_path / "legacy.py").write_text("frozen source\n")
    prior_protocol.write_text(json.dumps(old_protocol))
    prior_output.write_text(json.dumps({"decision": protocol["prior_decision"], "certificate_checks_passed": True,
        "effective_references_complete": True, "independent_oracles_complete": True}))
    prior_oracles.write_text(json.dumps({"registered_at_utc": "old", "oracles": oracles}))
    with gzip.open(screen._cell_path(prior_output, models[0].node, 2), "wt") as target:
        json.dump({"registered_at_utc": "old", "cell": rows[0]}, target)
    for name, value in (("ROOT", tmp_path), ("PRIOR_OUTPUT", prior_output),
                        ("PRIOR_PROTOCOL", prior_protocol), ("PRIOR_ORACLES", prior_oracles), ("EXTRA_SOURCES", ())):
        monkeypatch.setattr(screen, name, value)
    registered = screen.build_protocol(models)
    registered["registered_at_utc"] = "new"
    return screen, models, registered


def test_protocol_freezes_both_budgets_and_predecessor_source(saved_case):
    screen, models, protocol = saved_case
    screen.verify_protocol(protocol, models)
    for key in ("budget", "canonical_state_budget"):
        changed = deepcopy(protocol)
        changed[key] += 1
        with pytest.raises(ValueError, match="changed"):
            screen.verify_protocol(changed, models)
    (screen.ROOT / "legacy.py").write_text("changed source\n")
    with pytest.raises(ValueError, match="predecessor source changed"):
        screen.build_protocol(models)


def test_frozen_cells_and_oracles_require_the_original_registration_and_roster(saved_case):
    screen, models, protocol = saved_case
    assert len(screen.load_oracles(protocol)) == 1
    assert screen.load_prior_cell(protocol, models[0].node, 2)["states"][0]["queue"] == [3, 3, 0]
    changed = deepcopy(protocol)
    changed["prior_registered_at_utc"] = "different"
    with pytest.raises(ValueError, match="registration"):
        screen.load_oracles(changed)
    with pytest.raises(ValueError, match="registration"):
        screen.load_prior_cell(changed, models[0].node, 2)
    changed = deepcopy(protocol)
    changed["queue_panels"]["2"][0]["states"][0]["queue"] = [4, 3, 0]
    with pytest.raises(ValueError, match="roster"):
        screen.load_prior_cell(changed, models[0].node, 2)
