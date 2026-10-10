"""Certificate routing, retained failures and three-arm development gates."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import certificate_routed_development_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


def expected_key(state, model):
    return runtime.reference_key(selection_from_record(state["reference"]), state["queue"], len(model.options))


def refresh(protocol, row, model):
    snapshot = model.snapshot()
    for state in row["states"]:
        expected = expected_key(state, model)
        state["screen_metrics"] = summary.audit_candidate(
            state["screen_result"], state, protocol, snapshot, 4, expected)[0]
        for label in summary.LABELS:
            state["metrics"][label] = runtime._combine_metrics(
                summary.audit_result(result, state, protocol, snapshot, 4, label, expected)[0]
                for result in summary._repeat_results(state, label, protocol["repeats"]))
    row["screen_quality_passed"], row["screen_failures"] = summary.screen_audit(protocol, row, snapshot)
    for item in row["warmups"]:
        state = row["states"][item["state_index"]]
        item["metrics"] = summary.audit_result(item["result"], state, protocol, snapshot, 4,
                                                item["label"], expected_key(state, model))[0]


@pytest.fixture
def case():
    pairs = (((1, 0, 0), 5.), ((0, 1, 0), 2.), ((2, 1, 0), 5.),
             ((1, 2, 0), 3.), ((3, 0, 0), 2.), ((0, 3, 0), 3.))
    model = FrameModel(node="synthetic", scenario_id="tiny", lanes=4,
        workloads=("a", "b", "c"), scales=(1., 1., 1.),
        options=tuple(FrameOption(str(i), req, duration, ((), (), ())) for i, (req, duration) in enumerate(pairs)))
    protocol = {"experiment": "synthetic_certificate_routed", "registered_at_utc": "test", "models": [model.snapshot()],
        "lanes": [4], "repeats": 6, "warmups_per_label": 2, "ordinary_per_cell": 8,
        "initial_lp_budget": 3, "lp_node_budget": 64, "integer_state_budget": 8192,
        "long_lp_budget": 4096, "control_integer_state_budget": 100000,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "cpu": 0, "score_gain_limit": .95, "state_ratio_limit": 1.05,
        "nonregression_limit": 1.05, "trigger_long_ratio_limit": 1.0,
        "timing_indices": {"synthetic:4": [0, 1, 2, 3]}, "limitations": "Synthetic only.",
        "queue_panels": {"4": [{"panel": "tiny", "cohort": "development", "seed": None,
            "states": [{"state_id": "ordinary", "queue": [0, 0, 0]},
                       {"state_id": "other_trigger", "queue": [1, 6, 0]},
                       {"state_id": "score_a", "queue": [4, 6, 0]},
                       {"state_id": "score_b", "queue": [6, 4, 0]}]}]}}
    states = []
    for registered in protocol["queue_panels"]["4"][0]["states"]:
        queue = tuple(registered["queue"])
        reference = runtime.select_canonical_count_space_bnb_frame(model.options, queue, 4, 4096)
        expected = runtime.reference_key(reference, queue, len(model.options))
        reference_metrics = runtime._metrics()
        runtime._record(reference_metrics, reference, model, queue, 4, 4096, expected)
        historical = asdict(select_strict_score_proof_canonical_completion_frame(model.options, queue, 4, 3, 100000))
        probe = asdict(select_resumable_probe_frame(model.options, queue, 4, 3, 64, 8192))
        candidate = asdict(select_certificate_routed_frame(model.options, queue, 4, 3, 64, 8192))
        long = asdict(select_strict_score_proof_canonical_completion_frame(model.options, queue, 4, 4096, 100000))
        assert all(result["selection"]["canonical_exact"] for result in (probe, candidate, long))
        states.append({**registered, "panel": "tiny", "cohort": "development", "seed": None,
            "reference": asdict(reference), "reference_metrics": reference_metrics, "reference_complete": True,
            "oracle": None, "effective_reference_complete": True, "effective_reference_source": "full_K4096",
            "expected_initial_lp": {"selection": historical["baseline_selection"], "preprocessing": historical["preprocessing"],
                                    "capacity_propagation": historical["capacity_propagation"]},
            "expected_control_result": historical, "expected_probe_result": deepcopy(probe),
            "score_unproved": not historical["baseline_selection"]["score_exact"],
            "previous_integer_trigger": registered["state_id"] != "ordinary",
            "screen_result": deepcopy(candidate), "first_results": {"candidate": candidate, "probe": probe, "long": long},
            "repeat_mismatches": {label: [] for label in summary.LABELS}, "metrics": {},
            "wall_ns": {"candidate": [80] * 6, "probe": [100] * 6, "long": [100] * 6},
            "cpu_ns": {"candidate": [70] * 6, "probe": [90] * 6, "long": [90] * 6}})
    row = {"node": "synthetic", "lanes": 4, "physical_lanes": 4, "option_count": len(model.options),
        "registered_at_utc": "test", "states": states,
        **{key: {"cpu_affinity": [0]} for key in ("runtime_start", "runtime_before_warmup", "runtime_after_warmup", "runtime_end")},
        "warmups": [{"label": label, "state_index": 0, "result": deepcopy(states[0]["first_results"][label])}
                    for _ in range(2) for label in summary.LABELS]}
    refresh(protocol, row, model)
    return protocol, row, model


def test_all_groups_pass_and_cumulative_lp_is_charged_once(case):
    protocol, row, _ = case
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "CERTIFICATE_ROUTED_DEVELOPMENT_ADVANCE"
    assert result["screen_selector_calls"] == 4 and result["timed_selector_calls"] == 72
    assert result["warmup_calls"] == 6 and result["executed_selector_calls"] == 82
    assert result["executed_reference_calls"] == result["executed_oracle_calls"] == 0
    assert result["score_unproved"]["queue_count"] == 2
    assert result["other_trigger"]["queue_count"] == result["ordinary"]["queue_count"] == 1
    assert result["prior_integer_trigger"]["queue_count"] == 3
    screen = result["screen_work"]
    assert screen["search_work"]["lp_calls"] == sum(state["screen_result"]["final_lp"]["selection"]["lp_calls"] for state in row["states"])
    assert screen["search_work"]["lp_calls"] == screen["initial_lp_calls_embedded"] + screen["additional_lp_calls"]
    assert screen["score_proof"]["work"]["states_visited"] == 0
    expected = screen["search_work"]["lp_calls"] + sum(
        result["timed"]["work"][label]["search_work"]["lp_calls"] +
        result["warmup_work"][label]["search_work"]["lp_calls"] for label in summary.LABELS)
    assert result["executed_search_work"]["lp_calls"] == expected


@pytest.mark.parametrize("state_id,control,candidate_ns,control_ns,gate", (
    ("score_a", "probe", 112, 100, "score_unproved_probe_gain"),
    ("score_a", "probe", 106, 100, "score_unproved_probe_no_state_regression"),
    ("score_a", "long", 80, 50, "score_unproved_long_no_regression"),
    ("other_trigger", "probe", 106, 100, "other_trigger_probe_no_regression"),
    ("other_trigger", "long", 104, 50, "trigger_long_no_regression"),
    ("ordinary", "probe", 106, 100, "ordinary_probe_no_regression"),
    ("ordinary", "long", 80, 50, "ordinary_long_no_regression"),
))
def test_fixed_subgroup_gates_retain_regressions(case, state_id, control, candidate_ns, control_ns, gate):
    protocol, row, _ = case
    state = next(state for state in row["states"] if state["state_id"] == state_id)
    state["wall_ns"]["candidate"] = [candidate_ns] * 6
    state["wall_ns"][control] = [control_ns] * 6
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "CERTIFICATE_ROUTED_DEVELOPMENT_TIMING_HOLD"
    assert result["quality_gate_passed"] and not result["timing_gates"][gate]
    assert result["timed_queue_count"] == 4


def test_incomplete_score_keeps_all_calls_and_becomes_quality_hold(case):
    protocol, row, model = case
    protocol["lp_node_budget"] = 3
    for state in row["states"]:
        candidate = asdict(select_certificate_routed_frame(model.options, tuple(state["queue"]), 4, 3, 3, 8192))
        probe = asdict(select_resumable_probe_frame(model.options, tuple(state["queue"]), 4, 3, 3, 8192))
        state["screen_result"] = deepcopy(candidate)
        state["first_results"]["candidate"] = candidate
        state["expected_probe_result"] = deepcopy(probe)
        state["first_results"]["probe"] = probe
    for item in row["warmups"]:
        item["result"] = deepcopy(row["states"][0]["first_results"][item["label"]])
    refresh(protocol, row, model)
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "CERTIFICATE_ROUTED_DEVELOPMENT_QUALITY_HOLD"
    assert result["certificate_checks_passed"]
    assert result["timed_selector_calls"] == 72 and result["executed_selector_calls"] == 82
    assert len([failure for failure in result["quality_failures"] if failure["label"] == "candidate"]) == 14
    assert row["states"][2]["screen_result"]["probe"] is None


def test_resumed_probe_is_audited_against_final_paid_snapshot(case):
    protocol, row, model = case
    protocol["lp_node_budget"] = 5
    state = row["states"][2]
    candidate = asdict(select_certificate_routed_frame(model.options, tuple(state["queue"]), 4, 3, 5, 8192))
    assert candidate["probe_stage"] == "resumed"
    assert not candidate["initial_lp"]["selection"]["score_exact"]
    assert candidate["final_lp"]["selection"]["score_exact"]
    _, errors, quality = summary.audit_candidate(candidate, state, protocol, model.snapshot(), 4, expected_key(state, model))
    assert errors == [] and quality
    assert candidate["probe"]["score_proof"]["states_visited"] == 0
    candidate["probe"]["baseline_selection"] = deepcopy(candidate["initial_lp"]["selection"])
    _, errors, _ = summary.audit_candidate(candidate, state, protocol, model.snapshot(), 4, expected_key(state, model))
    assert errors


@pytest.mark.parametrize("fault", ("strict_proof", "initial_history", "unpaid_prefix", "wrong_stage", "probe_history", "subgroup"))
def test_routing_history_and_paid_work_violations_are_rejected(case, fault):
    protocol, row, model = case
    state = row["states"][1 if fault == "strict_proof" else 2]
    candidate = state["screen_result"]
    if fault == "strict_proof":
        candidate["probe"]["score_proof"]["states_visited"] = 1
        candidate["probe"]["completion"]["state_budget"] -= 1
    elif fault == "initial_history":
        candidate["initial_lp"]["capacity_propagation"]["region_visits"] += 1
    elif fault == "unpaid_prefix":
        candidate["final_lp"]["selection"]["lp_calls"] = 1
    elif fault == "wrong_stage":
        candidate["probe_stage"] = "initial"
    elif fault == "probe_history":
        state["first_results"]["probe"]["final_lp"]["capacity_propagation"]["region_visits"] += 1
    else:
        state["score_unproved"] = False
    refresh(protocol, row, model)
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "INVALID_EXECUTION"
    assert result["accounting_failures"]


def test_differing_valid_repeat_is_retained_and_blocks_timing(case):
    protocol, row, model = case
    state = row["states"][2]
    changed = deepcopy(state["first_results"]["candidate"])
    for selected in (changed["selection"], changed["final_lp"]["selection"]):
        selected["candidate_evaluations"] += 1
    state["repeat_mismatches"]["candidate"].append({"repeat_index": 2, "result": changed})
    refresh(protocol, row, model)
    result = summary.summarize(protocol, [row])
    assert result["decision"] == "INVALID_TIMING_EXECUTION"
    assert result["quality_gate_passed"] and not result["repeat_paths_stable"]
    expected = sum(state["first_results"]["candidate"]["final_lp"]["selection"]["candidate_evaluations"] * 6 for state in row["states"])
    assert result["timed"]["work"]["candidate"]["search_work"]["candidate_evaluations"] == expected + 1


@pytest.mark.parametrize("fault", ("missing_repeat", "missing_warmup", "missing_state"))
def test_missing_registered_measurements_cannot_disappear(case, fault):
    protocol, row, _ = case
    if fault == "missing_repeat":
        row["states"][0]["wall_ns"]["long"].pop()
    elif fault == "missing_warmup":
        row["warmups"].pop()
    else:
        row["states"].pop()
    with pytest.raises(ValueError):
        summary.summarize(protocol, [row])
