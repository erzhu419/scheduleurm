"""Development gates and cumulative accounting on two tiny synthetic queues."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import resumable_probe_development_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


def refresh(protocol, row, model):
    for state in row["states"]:
        expected = runtime.reference_key(selection_from_record(state["reference"]), state["queue"], len(model.options))
        state["screen_metrics"] = summary.audit_candidate(state["screen_result"], state, protocol, model.snapshot(), 4, expected)[0]
        for label in summary.LABELS:
            audit = summary.audit_candidate if label == "candidate" else summary._audit_control
            state["metrics"][label] = runtime._combine_metrics(
                audit(result, state, protocol, model.snapshot(), 4, expected)[0]
                for result in summary._repeat_results(state, label, protocol["repeats"]))
    row["screen_quality_passed"], row["screen_failures"] = summary.screen_audit(protocol, row, model.snapshot())
    for item in row["warmups"]:
        state = row["states"][item["state_index"]]
        expected = runtime.reference_key(selection_from_record(state["reference"]), state["queue"], len(model.options))
        audit = summary.audit_candidate if item["label"] == "candidate" else summary._audit_control
        item["metrics"] = audit(item["result"], state, protocol, model.snapshot(), 4, expected)[0]


@pytest.fixture
def case():
    pairs = (((1, 0, 0), 5.), ((0, 1, 0), 2.), ((2, 1, 0), 5.),
             ((1, 2, 0), 3.), ((3, 0, 0), 2.), ((0, 3, 0), 3.))
    model = FrameModel(node="synthetic", scenario_id="tiny", lanes=4,
        workloads=("a", "b", "c"), scales=(1., 1., 1.),
        options=tuple(FrameOption(str(i), req, duration, ((), (), ())) for i, (req, duration) in enumerate(pairs)))
    protocol = {"experiment": "synthetic_probe", "registered_at_utc": "test", "models": [model.snapshot()],
        "lanes": [4], "repeats": 6, "warmups_per_label": 2, "ordinary_per_cell": 8,
        "initial_lp_budget": 3, "lp_node_budget": 64, "integer_state_budget": 0,
        "control_integer_state_budget": 100000, "reference_budget": 4096, "oracle_state_budget": 5000000,
        "cpu": 0, "trigger_ratio_limit": .90, "trigger_state_ratio_limit": 1.05, "ordinary_ratio_limit": 1.05,
        "timing_indices": {"synthetic:4": [0, 1]}, "limitations": "Synthetic only.",
        "queue_panels": {"4": [{"panel": "tiny", "cohort": "development", "seed": None,
            "states": [{"state_id": "ordinary", "queue": [0, 0, 0]}, {"state_id": "trigger", "queue": [1, 6, 0]}]}]}}
    states = []
    for registered in protocol["queue_panels"]["4"][0]["states"]:
        queue = tuple(registered["queue"])
        reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, 4, 4096)
        expected = runtime.reference_key(reference, queue, len(model.options))
        reference_metrics = runtime._metrics()
        runtime._record(reference_metrics, reference, model, queue, 4, 4096, expected)
        control = asdict(select_strict_score_proof_canonical_completion_frame(model.options, queue, 4, 3, 100000))
        candidate = asdict(select_resumable_probe_frame(model.options, queue, 4, 3, 64, 0))
        assert reference.canonical_exact and control["selection"]["canonical_exact"] and candidate["selection"]["canonical_exact"]
        states.append({**registered, "panel": "tiny", "cohort": "development", "seed": None,
            "reference": asdict(reference), "reference_metrics": reference_metrics, "reference_complete": True,
            "oracle": None, "effective_reference_complete": True, "effective_reference_source": "full_K4096",
            "expected_initial_lp": {"selection": control["baseline_selection"], "preprocessing": control["preprocessing"],
                                    "capacity_propagation": control["capacity_propagation"]},
            "expected_control_result": deepcopy(control), "previous_integer_trigger": registered["state_id"] == "trigger",
            "screen_result": deepcopy(candidate), "first_results": {"candidate": candidate, "control": control},
            "repeat_mismatches": {label: [] for label in summary.LABELS}, "metrics": {},
            "wall_ns": {"candidate": [80] * 6, "control": [100] * 6},
            "cpu_ns": {"candidate": [70] * 6, "control": [90] * 6}})
    row = {"node": "synthetic", "lanes": 4, "physical_lanes": 4, "option_count": len(model.options),
        "registered_at_utc": "test", "states": states,
        **{key: {"cpu_affinity": [0]} for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")},
        "warmups": [{"label": label, "state_index": 0, "result": deepcopy(states[0]["first_results"][label])}
                    for _ in range(2) for label in summary.LABELS]}
    refresh(protocol, row, model)
    return protocol, row, model


def test_equal_quality_gain_advances_and_lp_prefix_is_charged_once(case):
    protocol, row, _ = case
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "RESUMABLE_PROBE_DEVELOPMENT_ADVANCE"
    assert result["screen_selector_calls"] == 2 and result["timed_selector_calls"] == 24 and result["warmup_calls"] == 4
    assert result["executed_selector_calls"] == 30
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["screen_work"]["search_work"]["lp_calls"] == 3 + 7
    assert result["screen_work"]["initial_lp_calls_embedded"] == 6
    assert result["screen_work"]["additional_lp_calls"] == 4
    assert result["prior_integer_trigger"]["queue_count"] == result["ordinary"]["queue_count"] == 1
    expected = result["screen_work"]["search_work"]["lp_calls"] + sum(
        result["timed"]["work"][label]["search_work"]["lp_calls"] +
        result["warmup_work"][label]["search_work"]["lp_calls"] for label in summary.LABELS)
    assert result["executed_search_work"]["lp_calls"] == expected


@pytest.mark.parametrize("group,ratio", (("trigger", .95), ("trigger", 1.06), ("ordinary", 1.06)))
def test_each_fixed_timing_gate_can_hold_despite_other_subset_gain(case, group, ratio):
    protocol, row, _ = case
    state = next(state for state in row["states"] if state["state_id"] == group)
    state["wall_ns"]["candidate"] = [int(ratio * 100)] * 6
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "RESUMABLE_PROBE_DEVELOPMENT_TIMING_HOLD"
    assert result["quality_gate_passed"] and not result["development_timing_gate_passed"]


def test_failed_screen_retains_every_preregistered_timing_call(case):
    protocol, row, model = case
    state = row["states"][1]
    # Same supported selector with no additional LP allowance honestly fails to close.
    protocol["lp_node_budget"] = 3
    for current in row["states"]:
        result = asdict(select_resumable_probe_frame(model.options, tuple(current["queue"]), 4, 3, 3, 0))
        current["screen_result"] = deepcopy(result)
        current["first_results"]["candidate"] = result
    for item in row["warmups"]:
        if item["label"] == "candidate":
            item["result"] = deepcopy(row["states"][0]["screen_result"])
    refresh(protocol, row, model)
    assert not state["screen_result"]["selection"]["canonical_exact"]
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "RESUMABLE_PROBE_DEVELOPMENT_QUALITY_HOLD"
    assert result["certificate_checks_passed"]
    assert result["timed_selector_calls"] == 24 and result["executed_selector_calls"] == 30
    assert len(result["quality_failures"]) == 7


def test_valid_differing_repeat_is_preserved_and_invalidates_timing(case):
    protocol, row, model = case
    state = row["states"][1]
    changed = deepcopy(state["first_results"]["candidate"])
    # Different feasible cumulative work is still a different timed execution path.
    for selected in (changed["selection"], changed["final_lp"]["selection"]):
        selected["candidate_evaluations"] += 1
    state["repeat_mismatches"]["candidate"].append({"repeat_index": 2, "result": changed})
    refresh(protocol, row, model)
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
    assert result["quality_gate_passed"] and not result["repeat_paths_stable"]
    assert result["timed"]["work"]["candidate"]["search_work"]["candidate_evaluations"] == sum(
        s["first_results"]["candidate"]["final_lp"]["selection"]["candidate_evaluations"] * 6 for s in row["states"]) + 1


@pytest.mark.parametrize("fault", ("initial_history", "unpaid_prefix", "preprocessing_restart", "probe_budget", "control_history"))
def test_accounting_or_history_failure_invalidates_development(case, fault):
    protocol, row, model = case
    state = row["states"][1]
    result = state["screen_result"]
    if fault == "initial_history":
        result["initial_lp"]["capacity_propagation"]["region_visits"] += 1
    elif fault == "unpaid_prefix":
        result["final_lp"]["selection"]["lp_calls"] = 1
    elif fault == "preprocessing_restart":
        result["final_lp"]["preprocessing"]["grouping_option_probes"] *= 2
    elif fault == "probe_budget":
        result["integer_state_budget"] += 1
    else:
        state["first_results"]["control"]["completion"]["memo_hits"] += 1
    refresh(protocol, row, model)
    summarized = summary.summarize(protocol, [row])
    assert summarized["decision"] == "INVALID_EXECUTION"
    assert summarized["accounting_failures"]


@pytest.mark.parametrize("fault", ("missing_state", "missing_repeat", "missing_warmup", "wrong_subset", "duplicate_override"))
def test_missing_or_changed_registered_measurements_raise(case, fault):
    protocol, row, _ = case
    if fault == "missing_state":
        row["states"].pop()
    elif fault == "missing_repeat":
        row["states"][0]["wall_ns"]["candidate"].pop()
    elif fault == "missing_warmup":
        row["warmups"].pop()
    elif fault == "wrong_subset":
        protocol["timing_indices"]["synthetic:4"] = [0]
    else:
        state = row["states"][0]
        changed = deepcopy(state["first_results"]["candidate"])
        changed["status"] = "changed"
        state["repeat_mismatches"]["candidate"] = [{"repeat_index": 2, "result": changed}] * 2
    with pytest.raises(ValueError):
        summary.summarize(protocol, [row])
