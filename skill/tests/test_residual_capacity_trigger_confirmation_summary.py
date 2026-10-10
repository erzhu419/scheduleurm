"""Locked fresh triggers, independent axes, retained-prefix fees and controls."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from algorithm.experiments.aligned_hardware_residual_capacity_trigger_confirmation import validation_protocol
from algorithm.experiments.residual_capacity_trigger_confirmation_summary import summarize
from algorithm.residual_capacity_frontier_frame_candidates import ResidualCapacityCompletionWork
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from skill.tests.test_axis_capacity_frontier_development_summary import axis_route
from skill.tests.test_frontier_trigger_discovery_summary import (
    records as discovery_records, screen_round as ordinary_screen_round, validation_state,
)


def residual_state(metadata, *, active=True, proved=False):
    state = validation_state(metadata, proved=proved)
    state["source"] = metadata.get("source", "fresh_confirmation")
    state["previous_residual_routed"] = None
    state["prior_routed"] = {}
    for label, field in (("old", "old_routed"), ("new", "routed")):
        result = state[field]
        if proved:
            state["prior_routed"][label] = deepcopy(result)
            continue
        if active:
            for snapshot in state["traces"][label].values():
                snapshot["selection"]["frontier_regions_remaining"] = 2
            result.update(initial_lp=deepcopy(state["traces"][label]["144"]),
                          final_lp=deepcopy(state["traces"][label]["256"]))
            exported = result["frontier_evidence"]
            exported.update(raw_region_count=2, live_region_count=2)
            second = deepcopy(exported["regions"][0])
            second["lower"] = [0]
            exported["regions"].append(second)
            exported["export_work"].update(heap_entries_read=2, group_ids_copied=2, requirement_rows_copied=2,
                requirement_coordinates_copied=2, box_count_values_copied=4, exact_duration_positions_copied=2,
                representative_options_copied=2, fraction_pairs_copied=5)
        prior = axis_route(result, dp=active)
        current = deepcopy(prior)
        capacity = current["probe"]["score_proof"]["capacity"]
        capacity["merged_values_copied"] = 3 * len(capacity["bounds"])
        queue = state["queue"][0]
        for bound in capacity["bounds"]:
            bound["exact_slot_numerators"] = [0, queue, 2 * queue]
        if active:
            current["probe"]["completion"] = asdict(ResidualCapacityCompletionWork(
                "COMPLETE", True, True, 8184, states_visited=5, thresholds_checked=1, score_fraction=(2 * queue, 1),
                prefix_states=3, dfs_states_visited=2, prefix_reads=3, capacity_queries=1,
                tables_reused=1, coverage_coordinate_reads=2))
            prior["probe"]["completion"] = asdict(ScoreFaceCompletionWork(
                "STATE_BUDGET_EXHAUSTED", True, False, 8184, states_visited=8184,
                thresholds_checked=1, score_fraction=(2 * queue, 1)))
            prior["probe"]["selection"] = deepcopy(prior["probe"]["score_certified_selection"])
            prior.update(selection=deepcopy(prior["probe"]["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
        state["prior_routed"][label], state[field] = prior, current
    return state


def screen_round(protocol, index, *, active=True, proved_seeds=()):
    record = ordinary_screen_round(protocol, index, proved_seeds=proved_seeds)
    for state in record["states"]:
        if active and not state["stage"]["selection"]["score_exact"]:
            state["stage"]["selection"]["frontier_regions_remaining"] = 2
    return record


def validation_records(protocol, discovery, *, active=True):
    models = [SimpleNamespace(node=model["node"], options=tuple(SimpleNamespace(**option) for option in model["options"]))
              for model in protocol["models"]]
    validation = validation_protocol(protocol, discovery, models)
    validation["registered_at_utc"] = "locked_validation"
    runtime, rows = {"cpu_affinity": [protocol["cpu"]], "gc_enabled": True}, []
    for cell in validation["cells"]:
        states = []
        for registered in cell["states"]:
            state = residual_state(registered, active=active,
                proved=registered["source"] == "fresh_confirmation" and cell["node"] not in registered["trigger_nodes"])
            if registered["source"] != "fresh_confirmation":
                state["previous_residual_routed"] = {"old": deepcopy(state["old_routed"]), "new": deepcopy(state["routed"])}
            states.append(state)
        fresh = sum(state["source"] == "fresh_confirmation" for state in states)
        borrowed = len(states) - fresh
        rows.append({"node": cell["node"], "lanes": 2, "physical_lanes": 2, "option_count": 1,
            "registered_at_utc": "locked_validation", "runtime_start": deepcopy(runtime), "runtime_end": deepcopy(runtime),
            "states": states, "new_reference_calls": fresh, "new_reference_oracle_calls": 0,
            "borrowed_reference_count": borrowed, "borrowed_native_snapshots": 8 * borrowed,
            "borrowed_prior_axis_count": 2 * borrowed, "new_prior_axis_selector_calls": 2 * fresh,
            "new_residual_selector_calls": 2 * len(states), "executed_native_advance_calls": 8 * fresh,
            "fresh_native_search_constructions": 2 * fresh})
    return validation, rows


@pytest.fixture
def records():
    protocol, discovery, _, _ = deepcopy(discovery_records.__wrapped__())
    controls = []
    for model in protocol["models"]:
        controls.append({"node": model["node"], "lanes": 2, "states": [
            {"state_id": f"control:q{queue}", "queue": [queue], "source": cohort, "cohort": cohort,
             "panel": cohort, "historical_regression": False, "caps": [1, 2, 144, 256],
             "borrowed_from": "residual_capacity_development", "borrowed_index": index,
             "source_state_id": f"previous:q{queue}", "source_record_source": cohort}
            for index, (queue, cohort) in enumerate(((10, "failure_development"), (11, "axis_failure_development")))]})
    protocol.update(min_residual_queues=2, min_residual_seeds=2)
    protocol["excluded_static_queues"]["2"] = [[10], [11]]
    protocol["validation_template"].update(control_cells=controls,
        failure_development_keys=[{"node": model["node"], "lanes": 2, "queue": [10]} for model in protocol["models"]],
        axis_failure_development_keys=[{"node": model["node"], "lanes": 2, "queue": [11]} for model in protocol["models"]],
        reference_source={"path": "frozen_residual", "registered_at_utc": "previous_residual"})
    protocol.pop("reference_budget")
    protocol.pop("oracle_state_budget")
    protocol["prior_decisions"]["axis_capacity_development"] = "AXIS_CAPACITY_FRONTIER_DEVELOPMENT_HOLD"
    protocol["prior_decisions"]["residual_capacity_development"] = "RESIDUAL_CAPACITY_FRONTIER_DEVELOPMENT_ADVANCE"
    protocol["validation_template"]["prior_decisions"] = deepcopy(protocol["prior_decisions"])
    discovery["rounds"] = [screen_round(protocol, 0)]
    validation, rows = validation_records(protocol, discovery)
    return protocol, discovery, validation, rows


def test_fresh_reuse_closure_allows_incomplete_axis_baselines_and_charges_actual_phases(records):
    result = summarize(*records)
    assert result["decision"] == "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_ADVANCE" and result["prior_hold_retained"]
    assert result["trigger_queue_count"] == result["reused_table_queue_count"] == 2
    assert result["trigger_seed_count"] == result["reused_table_seed_count"] == 2
    assert result["fresh_state_count"] == result["control_state_count"] == result["new_reference_calls"] == 8
    assert result["borrowed_native_snapshots"] == 64 and result["borrowed_axis_router_count"] == 16
    assert result["validation_prior_axis_selector_calls"] == 16 and result["validation_residual_selector_calls"] == 32
    assert result["validation_native_advance_calls"] == 64 and result["executed_interface_calls"] == 120
    assert result["native_search_constructions"] == 72 and result["executed_search_work"]["lp_calls"] == 104
    assert result["residual_router_work"]["routed_new"]["axis_capacity"]["merged_values_copied"] == 48
    assert result["residual_router_work"]["routed_new"]["residual_capacity_completion"]["prefix_reads"] == 48
    assert result["residual_router_work"]["routed_new"]["shared_integer_states"] == 208
    assert len(result["axis_prior_nonclosures"]) == 32 and not result["routed_nonclosures"]
    assert result["reference_work"]["new"]["search_work"]["lp_calls"] == 8
    assert result["screen_new_reference_calls"] == result["screen_integer_probe_calls"] == 0
    assert not result["failures"]


@pytest.mark.parametrize("missing", ["catalogue", "fresh_queue"])
def test_all_locked_triggers_and_controls_must_keep_all_catalogues(records, missing):
    protocol, discovery, validation, rows = records
    if missing == "catalogue":
        rows.pop()
        validation["cells"].pop()
    else:
        rows[-1]["states"].pop(0)
        validation["cells"][-1]["states"].pop(0)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_discovery_cannot_continue_after_first_sufficient_full_round(records):
    protocol, discovery, validation, rows = records
    discovery["rounds"].append(screen_round(protocol, 1))
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


@pytest.mark.parametrize("proved_seeds", [(11, 12), (12,)])
def test_no_hits_or_one_seed_retains_all_controls_but_reports_insufficient_fresh_evidence(records, proved_seeds):
    protocol, discovery, _, _ = records
    discovery["rounds"] = [screen_round(protocol, index, proved_seeds=proved_seeds) for index in range(2)]
    discovery.update(stop_reason="screen_limit_reached", trigger_states=[
        {**deepcopy(state), "trigger_nodes": [model["node"] for model in protocol["models"]]}
        for panel in protocol["queue_panels"] if panel["seed"] not in proved_seeds for state in panel["states"]])
    validation, rows = validation_records(protocol, discovery)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "TRIGGER_INSUFFICIENT" and result["execution_checks_passed"]
    assert result["control_state_count"] == 8 and result["routed_quality_gate_passed"]
    assert not result["raw_trigger_sufficiency_passed"]


def test_controls_cannot_supply_fresh_residual_table_activation(records):
    protocol, discovery, _, _ = records
    discovery["rounds"] = [screen_round(protocol, 0, active=False)]
    validation, rows = validation_records(protocol, discovery, active=False)
    # Controls retain their frozen, active records independently of fresh activation.
    for cell, row in zip(validation["cells"], rows):
        for index, state in enumerate(row["states"]):
            if state["source"] != "fresh_confirmation":
                active = residual_state(cell["states"][index])
                active["previous_residual_routed"] = {"old": deepcopy(active["old_routed"]), "new": deepcopy(active["routed"])}
                row["states"][index] = active
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "TRIGGER_INSUFFICIENT" and result["raw_trigger_sufficiency_passed"]
    assert result["reused_table_queue_count"] == result["reused_table_seed_count"] == 0
    assert result["routed_quality_gate_passed"] and not result["failures"]


def test_nontriggering_catalogues_still_validate_every_locked_queue(records):
    protocol, discovery, _, _ = records
    for state in discovery["rounds"][0]["states"]:
        if state["node"] != "toy_a":
            state["stage"] = deepcopy(validation_state(state, proved=True)["traces"]["new"]["256"])
    for trigger in discovery["trigger_states"]:
        trigger["trigger_nodes"] = ["toy_a"]
    validation, rows = validation_records(protocol, discovery)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_ADVANCE"
    assert result["fresh_state_count"] == 8 and result["reused_table_queue_count"] == 2


def test_independent_reference_nonclosure_takes_priority_over_insufficiency(records):
    protocol, discovery, validation, rows = records
    protocol["min_residual_seeds"] = 3
    state = rows[0]["states"][0]
    state["reference"].update(exact=False, score_exact=False, canonical_exact=False, frontier_regions_remaining=1)
    state.update(reference_complete=False, effective_reference_complete=False, effective_reference_source=None)
    state["reference_metrics"].update(score_exact_count=0, canonical_exact_count=0, reference_comparable_count=0)
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INCOMPLETE_REFERENCE" and result["execution_checks_passed"]


def test_honest_canonical_exhaustion_is_method_hold_before_insufficiency(records):
    protocol, discovery, validation, rows = records
    protocol["min_residual_seeds"] = 3
    result_record = rows[0]["states"][0]["routed"]
    probe = result_record["probe"]
    probe["completion"].update(status="STATE_BUDGET_EXHAUSTED", complete=False, states_visited=8184, dfs_states_visited=8181)
    probe["selection"] = deepcopy(probe["score_certified_selection"])
    result_record.update(selection=deepcopy(probe["selection"]), status="FRONTIER_PROBE_INCOMPLETE")
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_HOLD"
    assert result["execution_checks_passed"] and result["predecessor_work_gate_passed"] and result["routed_nonclosures"]


def test_per_state_integer_growth_cannot_use_other_state_or_arm_savings(records):
    protocol, discovery, validation, rows = records
    state = rows[0]["states"][0]
    previous = state["prior_routed"]["new"]
    previous["probe"]["completion"] = asdict(ScoreFaceCompletionWork(
        "COMPLETE", True, True, 8184, states_visited=4, score_fraction=(12, 1)))
    previous["probe"]["selection"] = deepcopy(state["routed"]["selection"])
    previous.update(selection=deepcopy(previous["probe"]["selection"]), status="FRONTIER_PROBE_COMPLETE")
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "RESIDUAL_CAPACITY_TRIGGER_CONFIRMATION_HOLD" and result["execution_checks_passed"]
    assert result["predecessor_work_regressions"][0]["label"] == "new"


@pytest.mark.parametrize("fee", ["copy", "prefix", "coverage"])
def test_actual_retained_vector_prefix_and_coverage_fees_are_required(records, fee):
    protocol, discovery, validation, rows = records
    probe = rows[0]["states"][0]["routed"]["probe"]
    if fee == "copy":
        probe["score_proof"]["capacity"]["merged_values_copied"] = 0
    else:
        probe["completion"]["prefix_reads" if fee == "prefix" else "coverage_coordinate_reads"] = 0
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]


def test_known_controls_must_reproduce_saved_residual_outputs(records):
    protocol, discovery, validation, rows = records
    rows[0]["states"][-1]["routed"]["probe"]["completion"]["memo_hits"] += 1
    result = summarize(protocol, discovery, validation, rows)
    assert result["decision"] == "INVALID_EXECUTION" and result["failures"]
