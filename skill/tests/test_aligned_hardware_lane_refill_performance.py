"""Freeze the performance contrast and audit each actual free-lane decision."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from algorithm.experiments import aligned_hardware_lane_refill_performance as experiment
from algorithm.experiments import aligned_hardware_small_dimension_exact_native_common_cadence as preceding
from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import SCALAR_COSTS
from skill.tests.test_small_dimension_exact_native_common_cadence import previous_replay
from skill.tests.test_small_dimension_exact_native_decision_delay import registered_inputs


@pytest.fixture
def lane_inputs(previous_replay, monkeypatch):
    models, protocol, result = previous_replay
    monkeypatch.setattr(experiment, "ROOT", preceding.ROOT)
    monkeypatch.setattr(experiment, "PREVIOUS_OUTPUT", preceding.PREVIOUS_OUTPUT)
    monkeypatch.setattr(experiment, "SOURCES", ())
    return models, protocol, result


def test_performance_protocol_keeps_original_tapes_caps_and_utility_hold(lane_inputs):
    models, previous, result = lane_inputs
    protocol = experiment.build_protocol(models)
    for key in ("models", "lanes", "loads", "arrival_seeds", "arrival_horizon",
                "observation_end", "capacities", "tapes", "execution_order", "budgets",
                "initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget", "float_relative_tolerance", "cpu"):
        assert protocol[key] == previous[key]
    assert protocol["arrival_seeds"] == [41, 42, 43]
    assert len(protocol["tapes"]) == protocol["expected_tape_count"] == 24
    assert protocol["expected_trajectory_count"] == 72
    assert protocol["expected_group_count"] == 16
    assert tuple(protocol["labels"]) == ("sync_joint", "refill_joint", "refill_single")
    assert protocol["dispatch_rules"] == {
        "sync_joint": "synchronous", "refill_joint": "refill", "refill_single": "single"}
    assert protocol["idle_duration"] == 1.0
    assert protocol["minimum_pooled_flow_improvement"] == .01
    assert "computation_window_s" not in protocol
    assert "maximum_pooled_wall_ratio" not in protocol
    assert "maximum_group_wall_ratio" not in protocol
    assert "timing_orders" not in protocol
    assert protocol["prior_decisions"]["small_dimension_exact_native_decision_delay"] == result["decision"]
    assert result["decision"].endswith("UTILITY_HOLD")
    assert protocol["prior_decisions"]["older_utility"].endswith("UTILITY_HOLD")


@pytest.mark.parametrize("change", ("correctness", "source", "model"))
def test_failed_correctness_or_changed_native_source_catalogue_stops_registration(lane_inputs, change):
    models, _, result = lane_inputs
    if change == "correctness":
        result["certificate_checks_passed"] = False
        experiment.PREVIOUS_OUTPUT.write_text(json.dumps(result))
    elif change == "source":
        (experiment.ROOT / "whole_selector.py").write_text("changed native source\n")
    else:
        models = [replace(models[0], lanes=3), *models[1:]]
    with pytest.raises(ValueError):
        experiment.build_protocol(models)


@pytest.mark.parametrize("change", ("dispatch", "flow_gate"))
def test_registered_dispatch_and_flow_gate_cannot_change(lane_inputs, change):
    models, _, _ = lane_inputs
    protocol = experiment.build_protocol(models)
    protocol["registered_at_utc"] = "before performance outcomes"
    experiment.verify_protocol(protocol, models)
    if change == "dispatch":
        protocol["dispatch_rules"]["refill_joint"] = "synchronous"
    else:
        protocol["minimum_pooled_flow_improvement"] = 0.0
    with pytest.raises(ValueError, match="registered"):
        experiment.verify_protocol(protocol, models)


def test_all_three_actual_trajectories_precede_offline_audits(lane_inputs):
    models, _, _ = lane_inputs
    protocol = experiment.build_protocol(models)
    protocol["tapes"] = [protocol["tapes"][0]]
    tape = protocol["tapes"][0]
    timeline, originals_by_policy, cache_ids = [], {}, []
    selector = object()

    def simulate(model, arrivals, horizon, end, budget, selected, **kwargs):
        assert model.lanes == 8
        assert arrivals is tape["arrivals"]
        assert (horizon, end, budget, selected) == (600.0, 1200.0, 256, selector)
        label = kwargs["policy"]
        assert kwargs == {"policy": label, "dispatch_rule": protocol["dispatch_rules"][label]}
        timeline.append(("simulate", label))
        originals_by_policy[label] = [object()]
        return {"policy": label}, originals_by_policy[label]

    class Reference:
        def summary(self):
            return {"calls": 3, "oracle_calls": 0, "failures": 0,
                    "incomplete_count": 0, "work": {"search_work": {"lp_calls": 11}}}

    def audit(model, trajectory, originals, references, registered):
        assert registered is protocol and model.lanes == 8
        assert timeline[:3] == [("simulate", label) for label in experiment.LABELS]
        assert originals is originals_by_policy[trajectory["policy"]]
        cache_ids.append(id(references))
        references.setdefault(8, Reference())
        trajectory["audited"] = True
        timeline.append(("audit", trajectory["policy"]))

    dependencies = dict(simulate=simulate, audit=audit,
        selectors=lambda registered: {"candidate": selector}, provenance=lambda: {})
    row = experiment.run_cell(models[0], 8, protocol, dependencies=dependencies, progress=False)
    assert timeline == [(phase, label) for phase in ("simulate", "audit") for label in experiment.LABELS]
    assert len(set(cache_ids)) == 1
    run = row["runs"][0]
    assert run["arrival_count"] == len(tape["arrivals"])
    assert set(run["policies"]) == set(experiment.LABELS)
    assert all(output["audited"] for output in run["policies"].values())
    assert run["reference_audit"] == Reference().summary()


def test_free_lane_audit_separates_same_queue_references_and_preserves_all_fees(lane_inputs):
    models, _, _ = lane_inputs
    protocol = experiment.build_protocol(models)
    model = replace(models[0], lanes=8)
    frames = [{"queue": [4, 1, 2], "lanes_called": lanes, "tag": number,
               "start": float(number), "dispatch_time": float(number),
               "decision_delay_s": 0.0, "selector_wall_s": .01 * (number + 1)}
              for number, lanes in enumerate((8, 1, 8))]
    references, factories, received, original_tags = {}, [], [], {}

    class Reference:
        def __init__(self, local_model, registered):
            assert registered is protocol
            self.model, self.by_queue = local_model, {}
            factories.append(local_model.lanes)

    def audit(local_model, shadow, originals, reference, registered):
        assert registered is protocol and reference.model == local_model
        assert shadow["policy"] == "candidate"
        assert all(frame["lanes_called"] == local_model.lanes for frame in shadow["frames"])
        assert [original_tags[id(original)] for original in originals] == [frame["tag"] for frame in shadow["frames"]]
        assert all(frame["decision_delay_s"] == frame["selector_wall_s"] for frame in shadow["frames"])
        received.extend(originals)
        new_references = 0
        for frame in shadow["frames"]:
            queue = tuple(frame["queue"])
            if queue not in reference.by_queue:
                reference.by_queue[queue] = local_model.lanes
                new_references += 1
        count = len(originals)
        shadow["selector_metrics"] = {
            **dict.fromkeys(SCALAR_COSTS, 0), "calls": count, "canonical_exact_count": count,
            "operational_elapsed_s": sum(frame["decision_delay_s"] for frame in shadow["frames"]),
            "whole_work": {"search_work": {"lp_calls": 7 * count},
                "routing_preparation": {"root_snapshot_calls": count},
                "root_settle_work": {"rational_operations": 3 * count},
                "dual_face_work": {"rational_operations": 13 * count},
                "native_own_tight_work": {"event_snapshot_entries_copied": 5 * count},
                "small_dimension_lp_work": {"rational_operations": 11 * count},
                "score_proof": {"work": {"states_visited": 17 * count}},
                "max_shared_integer_states": 2 if local_model.lanes == 1 else 5}}
        shadow["decision_diagnostics"] = [{"frame": number, "status": str(frame["tag"])}
            for number, frame in enumerate(shadow["frames"])]
        shadow["reference_audit"] = {"calls": new_references, "oracle_calls": 0, "failures": 0,
            "incomplete_count": 0, "work": {"search_work": {"lp_calls": 19 * new_references}}}

    for number in range(2):
        originals = [object() for _ in frames]
        original_tags.update({id(original): frame["tag"] for original, frame in zip(originals, frames, strict=True)})
        trajectory = {"policy": "refill_joint", "frames": deepcopy(frames), "total_decision_delay_s": 0.0}
        experiment.audit_lane_calls(model, trajectory, originals, references, protocol,
                                    auditor=audit, reference_factory=Reference)
        assert trajectory["frames"] == frames
        assert trajectory["total_decision_delay_s"] == 0.0
        assert trajectory["selector_metrics"]["operational_elapsed_s"] == pytest.approx(.06)
        assert trajectory["selector_metrics"]["calls"] == 3
        assert trajectory["selector_metrics"]["whole_work"] == {
            "search_work": {"lp_calls": 21}, "routing_preparation": {"root_snapshot_calls": 3},
            "root_settle_work": {"rational_operations": 9}, "dual_face_work": {"rational_operations": 39},
            "native_own_tight_work": {"event_snapshot_entries_copied": 15},
            "small_dimension_lp_work": {"rational_operations": 33},
            "score_proof": {"work": {"states_visited": 51}}, "max_shared_integer_states": 5}
        assert trajectory["decision_diagnostics"] == [
            {"frame": frame["tag"], "status": str(frame["tag"]), "lanes_called": frame["lanes_called"]}
            for frame in frames]
        assert trajectory["reference_audit"]["calls"] == (2 if number == 0 else 0)
        assert trajectory["reference_audit"]["work"]["search_work"]["lp_calls"] == (38 if number == 0 else 0)
        assert all(any(actual is original for actual in received) for original in originals)
    assert factories == [1, 8]
    assert set(references) == {1, 8}
    assert references[1].by_queue == {(4, 1, 2): 1}
    assert references[8].by_queue == {(4, 1, 2): 8}
    assert len(received) == 6
