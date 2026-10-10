"""Conditional trigger discovery and validation on pure mocked paid records."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from algorithm.experiments.frontier_trigger_discovery_summary import summarize
from algorithm.experiments.aligned_hardware_queue_aware_canonical_development import selection_from_record
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import _metrics, _record, reference_key
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_dual_reuse_development_summary import new_trace
from skill.tests.test_frontier_score_proof_confirmation_summary import records as confirmation_records
from skill.tests.test_frontier_score_proof_development_summary import evidence
from skill.tests.test_late_score_proof_development_summary import late
from skill.tests.test_root_pruning_confirmation_summary import routed, selection, stage


def validation_state(metadata, *, proved=False):
    queue = metadata["queue"][0]
    state = {**deepcopy(metadata), "source": "fresh_confirmation", "historical_regression": False,
             "caps": [1, 2, 144, 256]}
    state["traces"] = {"old": {str(cap): stage(queue, cap, new=True, lp=2) for cap in state["caps"]},
                       "new": new_trace(state)}
    if not proved:
        for trace in state["traces"].values():
            for snapshot in trace.values():
                snapshot["selection"].update(exact=False, score_exact=False, canonical_exact=False,
                    upper_bound=float(2 * queue + 1), gap=1., epsilon=1 / queue, frontier_regions_remaining=1)
    for label, field in (("old", "old_routed"), ("new", "routed")):
        original = routed(state["traces"][label]["144"])
        if proved:
            state[field] = {**original, "frontier_evidence": None}
            continue
        original.update(final_lp=deepcopy(state["traces"][label]["256"]),
            selection=deepcopy(state["traces"][label]["256"]["selection"]), resumed=True, status="LP_RESUME_INCOMPLETE")
        result = late(original, 1)
        exported = evidence()
        exported["incumbent_score_fraction"] = [2 * queue, 1]
        exported["regions"][0]["bound_fraction"] = [2 * queue + 1, 1]
        result.update(probe_stage="frontier_late", status="FRONTIER_PROBE_COMPLETE", frontier_evidence=exported)
        result["probe"]["score_proof"]["thresholds_checked"] = 1
        state[field] = result
    reference = selection(queue, 4096, 1)
    selected = selection_from_record(reference)
    metric = _metrics()
    options = (SimpleNamespace(requirements=(1,), duration=1.),)
    _record(metric, selected, SimpleNamespace(options=options), (queue,), 2, 4096,
            reference_key(selected, (queue,), 1))
    state.update(reference=reference, reference_metrics=metric, reference_complete=True, oracle=None,
                 effective_reference_complete=True, effective_reference_source="full_K4096")
    return state


def screen_round(protocol, index, *, proved_seeds=()):
    states = []
    for panel in protocol["queue_panels"]:
        start = index * protocol["block_size"]
        for metadata in panel["states"][start:start + protocol["block_size"]]:
            snapshot = deepcopy(validation_state(metadata)["traces"]["new"]["256"])
            if panel["seed"] in proved_seeds:
                snapshot["selection"].update(score_exact=True, upper_bound=snapshot["selection"]["score"],
                    gap=0., epsilon=0., canonical_tie_regions_remaining=1)
            for model in protocol["models"]:
                states.append({**deepcopy(metadata), "node": model["node"], "lanes": protocol["lanes"],
                               "round_index": index, "stage": deepcopy(snapshot)})
    return {"round_index": index, "states": states}


def set_validation_counts(validation, rows):
    count = sum(len(row["states"]) for row in rows)
    validation.update(expected_state_count=count, expected_fresh_state_count=count,
        expected_failure_regression_count=0, expected_selector_calls=count * 2,
        expected_native_advance_calls=count * 8, expected_interface_calls=count * 10,
        expected_native_constructions=count * 4, expected_new_reference_calls=count)
    for row in rows:
        row.update(new_reference_calls=len(row["states"]), new_reference_oracle_calls=0,
                   new_old_routed_calls=len(row["states"]))


def validation_records(protocol, discovery):
    validation = deepcopy(protocol["validation_template"])
    validation.update(registered_at_utc="validation_fixed", discovery_registered_at_utc=protocol["registered_at_utc"],
        trigger_states=deepcopy(discovery["trigger_states"]), screen_stop_reason=discovery["stop_reason"], cells=[])
    runtime = {"cpu_affinity": [protocol["cpu"]], "gc_enabled": True}
    rows = []
    for model in protocol["models"]:
        states = [validation_state(trigger, proved=model["node"] not in trigger["trigger_nodes"])
                  for trigger in discovery["trigger_states"]]
        metadata = [{key: deepcopy(value) for key, value in state.items()
                     if key in ("state_id", "queue", "seed", "index", "panel", "cohort", "source", "historical_regression", "caps", "trigger_nodes")}
                    for state in states]
        validation["cells"].append({"node": model["node"], "lanes": 2, "states": metadata})
        rows.append({"node": model["node"], "lanes": 2, "physical_lanes": 2, "option_count": 1,
            "registered_at_utc": "validation_fixed", "states": states,
            "runtime_start": deepcopy(runtime), "runtime_end": deepcopy(runtime)})
    set_validation_counts(validation, rows)
    return validation, rows


@pytest.fixture
def records():
    template, _ = deepcopy(confirmation_records.__wrapped__())
    models = [{"node": "toy_" + suffix, "lanes": 2,
               "options": [{"requirements": [1], "duration": 1.}]} for suffix in "abcd"]
    panels = []
    for seed, queues in ((11, (6, 8)), (12, (7, 9))):
        panel = "seed_" + str(seed)
        panels.append({"seed": seed, "panel": panel, "states": [
            {"state_id": f"{panel}:q{queue}", "queue": [queue], "seed": seed, "index": index,
             "panel": panel, "cohort": "conditional_trigger_validation"}
            for index, queue in enumerate(queues)]})
    protocol = {key: deepcopy(template[key]) for key in ("cpu", "reference_budget", "initial_lp_budget",
        "lp_node_budget", "integer_state_budget", "oracle_state_budget", "prior_decisions", "limitations")}
    protocol.update(models=models, lanes=2, registered_at_utc="screen_fixed", panel_seeds=[11, 12],
        queues_per_panel=2, block_size=1, max_rounds=2, min_trigger_queues=2, min_trigger_seeds=2,
        queue_panels=panels, excluded_static_queues={"2": []}, expected_max_screen_calls=16,
        execution_sources={})
    protocol["validation_template"] = {key: deepcopy(protocol[key]) for key in ("cpu", "reference_budget",
        "initial_lp_budget", "lp_node_budget", "integer_state_budget", "oracle_state_budget", "prior_decisions",
        "limitations", "execution_sources", "models")}
    protocol["validation_template"].update(lanes=[2], historical_regression_keys=[], failure_regression_keys=[])
    triggers = [{**deepcopy(panel["states"][0]), "trigger_nodes": [model["node"] for model in models]}
                for panel in panels]
    runtime = {"cpu_affinity": [protocol["cpu"]], "gc_enabled": True}
    discovery = {"registered_at_utc": "screen_fixed", "runtime_start": deepcopy(runtime),
        "runtime_end": deepcopy(runtime), "rounds": [screen_round(protocol, 0)],
        "stop_reason": "two_seed_trigger_threshold", "trigger_states": deepcopy(triggers)}
    validation, rows = validation_records(protocol, discovery)
    return protocol, discovery, validation, rows


def test_complete_round_selects_two_seed_union_and_validates_every_catalogue(records):
    result = summarize(*records)
    assert result["decision"] == "FRONTIER_TRIGGER_VALIDATION_ADVANCE" and result["prior_hold_retained"]
    assert result["screen_state_count"] == result["validation_state_count"] == 8
    assert result["trigger_queue_count"] == result["trigger_seed_count"] == 2
    assert result["validation_audit"]["routed_quality_gate_passed"]
    assert result["screened_queue_count"] == result["screen_round_count"] * 2 == 2
    assert result["screen_work"]["search_work"]["lp_calls"] == 8
    assert result["validation_audit"]["executed_search_work"]["lp_calls"] == 48
    assert result["executed_search_work"]["lp_calls"] == 56
    assert result["screen_work"]["dual_reuse"]["certificates_saved"] == 8
    assert result["executed_dual_reuse_work"]["certificates_saved"] == 24
    assert result["executed_interface_calls"] == 88 and result["native_search_constructions"] == 40
    assert result["reference_work"]["new"]["search_work"]["lp_calls"] == result["new_reference_calls"] == 8
    assert result["screen_new_reference_calls"] == result["screen_new_reference_oracle_calls"] == result["screen_integer_probe_calls"] == 0
    assert result["validation_work"]["routed_new"]["shared_integer_states"] == 48
    assert result["executed_frontier_export_work"]["old"]["fraction_pairs_copied"] == 24
    assert result["executed_frontier_export_work"]["new"]["box_count_values_copied"] == 16
    assert "decision" not in result["validation_audit"] and "confirmation_gate_passed" not in result["validation_audit"]
    assert not result["failures"]


def test_catalogues_that_did_not_trigger_still_receive_both_cold_validation_methods(records):
    protocol, discovery, _, _ = records
    for record in discovery["rounds"][0]["states"]:
        if record["node"] != "toy_a":
            record["stage"] = deepcopy(validation_state(record, proved=True)["traces"]["new"]["256"])
    for trigger in discovery["trigger_states"]:
        trigger["trigger_nodes"] = ["toy_a"]
    validation, rows = validation_records(protocol, discovery)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "FRONTIER_TRIGGER_VALIDATION_ADVANCE" and not result["failures"]
    assert result["screen_score_unproved_count"] == 2 and result["validation_state_count"] == 8
    assert result["validation_selector_calls"] == 16


def test_no_strict_gain_is_needed_for_conditional_proof_validation(records):
    protocol, discovery, validation, rows = records
    for row in rows:
        for state in row["states"]:
            for cap in state["caps"]:
                state["traces"]["old"][str(cap)] = deepcopy(state["traces"]["new"][str(cap)])
                state["traces"]["old"][str(cap)].pop("dual_reuse")
            state["old_routed"] = deepcopy(state["routed"])
            for name in ("initial_lp", "final_lp"):
                state["old_routed"][name].pop("dual_reuse")
    result = summarize(protocol, discovery, validation, rows)
    assert result["validation_audit"]["execution_checks_passed"]
    assert not result["validation_audit"]["fresh_work_gate_passed"]
    assert result["decision"] == "FRONTIER_TRIGGER_VALIDATION_ADVANCE"


@pytest.mark.parametrize("missing", ["catalogue", "trigger_queue"])
def test_every_trigger_must_be_validated_on_all_four_registered_catalogues(records, missing):
    protocol, discovery, validation, rows = records
    if missing == "catalogue":
        rows.pop()
        validation["cells"].pop()
    else:
        rows[-1]["states"].pop()
        validation["cells"][-1]["states"].pop()
    set_validation_counts(validation, rows)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


@pytest.mark.parametrize("proved_seeds,trigger_seeds", [((11, 12), ()), ((12,), (11,))])
def test_full_screen_without_two_trigger_seeds_reports_insufficient_samples(records, proved_seeds, trigger_seeds):
    protocol, discovery, _, _ = records
    discovery["rounds"] = [screen_round(protocol, index, proved_seeds=proved_seeds) for index in range(2)]
    discovery.update(stop_reason="screen_limit_reached", trigger_states=[
        {**deepcopy(state), "trigger_nodes": [model["node"] for model in protocol["models"]]}
        for panel in protocol["queue_panels"] if panel["seed"] in trigger_seeds for state in panel["states"]])
    validation, rows = validation_records(protocol, discovery)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "TRIGGER_INSUFFICIENT" and result["execution_checks_passed"]
    assert result["trigger_seed_count"] == len(trigger_seeds)
    assert result["trigger_queue_count"] == 2 * len(trigger_seeds)
    assert result["validation_state_count"] == 8 * len(trigger_seeds) and not result["failures"]


@pytest.mark.parametrize("mutation", ["partial_round", "past_first_sufficient", "wrong_stop_reason", "screen_snapshot", "export_bound", "shared_budget", "candidate_reference"])
def test_discovery_and_real_validation_must_match_registered_evidence(records, mutation):
    protocol, discovery, validation, rows = records
    state = rows[0]["states"][0]
    if mutation == "partial_round":
        discovery["rounds"][0]["states"].pop()
    elif mutation == "past_first_sufficient":
        discovery["rounds"].append(screen_round(protocol, 1))
        discovery["trigger_states"].extend({**deepcopy(panel["states"][1]),
            "trigger_nodes": [model["node"] for model in protocol["models"]]} for panel in protocol["queue_panels"])
        validation, rows = validation_records(protocol, discovery)
    elif mutation == "wrong_stop_reason":
        discovery["stop_reason"] = validation["screen_stop_reason"] = "screen_limit_reached"
    elif mutation == "screen_snapshot":
        discovery["rounds"][0]["states"][0]["stage"]["selection"]["lp_matrix_coefficients"] += 1
    elif mutation == "export_bound":
        state["routed"]["frontier_evidence"]["regions"][0]["bound_fraction"] = [12, 1]
    elif mutation == "shared_budget":
        state["routed"]["probe"]["completion"]["state_budget"] = 8192
    else:
        state["effective_reference_source"] = "candidate"
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_honest_proof_exhaustion_is_a_quality_hold(records):
    protocol, discovery, validation, rows = records
    result = rows[0]["states"][0]["routed"]
    probe = result["probe"]
    baseline = deepcopy(probe["baseline_selection"])
    probe["score_proof"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8192)
    probe.update(selection=deepcopy(baseline), score_certified_selection=deepcopy(baseline),
        completion=asdict(ScoreFaceCompletionWork("SKIPPED_SCORE_UNPROVEN", False, False, 0)))
    result.update(selection=deepcopy(baseline), status="FRONTIER_PROBE_INCOMPLETE")
    summary = summarize(protocol, discovery, validation, rows)
    assert summary["decision"] == "FRONTIER_TRIGGER_VALIDATION_HOLD"
    assert summary["execution_checks_passed"] and not summary["failures"]


def test_independent_reference_nonclosure_is_not_rescued_from_trigger_actions(records):
    protocol, discovery, validation, rows = records
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]
    assert not result["failures"]
