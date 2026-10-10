"""Freeze the new clock without rewriting prior utility evidence or arrivals."""
from collections import Counter
import json

import pytest

from algorithm.experiments import aligned_hardware_small_dimension_exact_native_common_cadence as experiment
from algorithm.experiments import aligned_hardware_small_dimension_exact_native_decision_delay as preceding
from skill.tests.test_small_dimension_exact_native_decision_delay import registered_inputs


@pytest.fixture
def previous_replay(registered_inputs, monkeypatch):
    models, _ = registered_inputs
    protocol = preceding.build_protocol(models)
    monkeypatch.setattr(experiment, "ROOT", preceding.ROOT)
    monkeypatch.setattr(experiment, "PREVIOUS_OUTPUT", preceding.TAPE_OUTPUT)
    monkeypatch.setattr(experiment, "SOURCES", ())
    result = {key: True for key in ("execution_checks_passed", "runtime_checks_passed",
        "certificate_checks_passed", "references_complete", "canonical_action_gate_passed",
        "zero_trajectory_identity_passed", "integration_gate_passed")}
    result.update(decision="SMALL_DIMENSION_EXACT_NATIVE_DECISION_DELAY_UTILITY_HOLD",
        utility_gate_passed=False, flow_gain_established=False,
        prior_decisions={"older_utility": "INTEGER_CAPACITY_DECISION_DELAY_UTILITY_HOLD"})
    experiment.PREVIOUS_OUTPUT.write_text(json.dumps(result))
    experiment._protocol_path(experiment.PREVIOUS_OUTPUT).write_text(json.dumps(protocol))
    return models, protocol, result


def test_new_clock_reuses_all_tapes_orders_budgets_and_keeps_old_hold(previous_replay):
    models, previous, result = previous_replay
    protocol = experiment.build_protocol(models)
    for key in ("models", "capacities", "tapes", "timing_orders", "initial_lp_budget",
                "lp_node_budget", "integer_state_budget", "reference_budget", "oracle_state_budget"):
        assert protocol[key] == previous[key]
    assert protocol["computation_window_s"] == 1.0
    assert protocol["expected_trajectory_count"] == 312
    assert protocol["expected_reference_trajectory_count"] == 24
    assert protocol["expected_measured_trajectory_count"] == 288
    assert protocol["expected_speed_gate_count"] == 9
    assert protocol["maximum_pooled_wall_ratio"] == .95
    assert protocol["maximum_group_wall_ratio"] == 1.05
    assert protocol["prior_decisions"]["small_dimension_exact_native_decision_delay"] == result["decision"]
    for orders in protocol["timing_orders"].values():
        assert Counter(map(tuple, orders)) == {("candidate", "direct"): 3, ("direct", "candidate"): 3}


@pytest.mark.parametrize("change", ("source", "integration", "model"))
def test_changed_sources_or_failed_integration_stop_registration(previous_replay, change):
    models, _, result = previous_replay
    if change == "source":
        (experiment.ROOT / "whole_selector.py").write_text("changed source\n")
    elif change == "integration":
        result["integration_gate_passed"] = False
        experiment.PREVIOUS_OUTPUT.write_text(json.dumps(result))
    else:
        models = models[:-1]
    with pytest.raises(ValueError):
        experiment.build_protocol(models)


@pytest.mark.parametrize("key,value", (("computation_window_s", .1),
    ("maximum_pooled_wall_ratio", 1.0), ("maximum_group_wall_ratio", 1.1)))
def test_clock_and_cost_gates_cannot_change_after_registration(previous_replay, key, value):
    models, _, _ = previous_replay
    protocol = experiment.build_protocol(models)
    protocol["registered_at_utc"] = "before outcomes"
    experiment.verify_protocol(protocol, models)
    protocol[key] = value
    with pytest.raises(ValueError, match="registered"):
        experiment.verify_protocol(protocol, models)


def test_reference_is_normative_and_every_audit_follows_all_twelve_measurements(previous_replay):
    models, _, _ = previous_replay
    protocol = experiment.build_protocol(models)
    protocol["tapes"] = protocol["tapes"][:1]
    timeline = []

    def simulate(model, tape, horizon, end, budget, selector, window, policy):
        assert (model.lanes, horizon, end, window) == (8, 600., 1200., 1.)
        assert budget == (4096 if policy == "full_reference" else 256)
        timeline.append(("simulate", policy))
        return {"policy": policy}, [object()]

    class Reference:
        def __init__(self, model, registered):
            assert len(timeline) == 13
            timeline.append(("reference_audit",))

        def summary(self):
            return {"calls": 0}

    def audit(model, trajectory, original, reference, registered):
        assert len(original) == 1
        assert sum(entry[0] == "simulate" for entry in timeline) == 13
        timeline.append(("audit", trajectory["policy"]))
        trajectory["audited"] = True

    dependencies = dict(simulate=simulate, reference=Reference, audit=audit,
        selectors=lambda registered: dict.fromkeys(("full_reference", *experiment.LABELS), object()),
        provenance=lambda: {})
    row = experiment.run_cell(models[0], 8, protocol, dependencies=dependencies, progress=False)
    assert timeline[0] == ("simulate", "full_reference")
    assert timeline[13] == ("reference_audit",)
    assert Counter(label for _, label in timeline[1:13]) == {"candidate": 6, "direct": 6}
    run = row["runs"][0]
    assert run["reference"]["audited"]
    assert len(run["measured"]) == 6
    assert all(output["audited"] for block in run["measured"] for output in block["policies"].values())
