"""Check paired inputs, deferred audits, and seed-level decision-delay readout."""
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_decision_delay as runner
from algorithm.experiments import aligned_hardware_integer_capacity_development as kernels
from algorithm.experiments import aligned_hardware_frame_experiment as arrivals
from algorithm.experiments import integer_capacity_decision_delay_replay as replay
from algorithm.experiments import integer_capacity_decision_delay_summary as summary
from algorithm.experiments import integer_capacity_trajectory_replay as legacy
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_decision_delay_replay import simulate_delayed_trajectory


def _model(node="tiny", duration=2.0):
    return FrameModel(
        node=node, scenario_id="paired_delay_test", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (1, 0, 0), duration, ((duration,), (), ())),),
    )


@pytest.fixture
def registered(tmp_path, monkeypatch):
    models = tuple(_model(f"node{i}", float(i + 2)) for i in range(4))
    source = tmp_path / "selector.py"
    source.write_text("frozen selector\n")
    previous = tmp_path / "trajectory_protocol.json"
    previous.write_text(json.dumps({
        "models": [model.snapshot() for model in models],
        "execution_sources": {"selector.py": source.read_text()},
        "capacities": [
            {"node": model.node, "lanes": lanes, "load": load,
             "certificate": {"arrival_rates": [load * lanes / model.options[0].duration, 0, 0]}}
            for model in models for lanes in (4, 8, 16) for load in (0.8, 0.95)],
        "search_counters": ["lp_calls"], "capacity_counters": ["region_visits"],
    }))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TRAJECTORY_PROTOCOL", previous)
    monkeypatch.setattr(runner, "TRAJECTORY_OUTPUT", tmp_path / "trajectory.json")
    monkeypatch.setattr(runner, "EXTRA_SOURCES", ())
    monkeypatch.setattr(runner, "require_trajectory_integration", lambda: {
        "decision": "INTEGER_CAPACITY_TRAJECTORY_UTILITY_HOLD", "integration_gate_passed": True,
        "utility_gate_passed": False, "flow_gain_established": False})
    monkeypatch.setattr(arrivals, "poisson_arrivals", lambda rates, horizon, seed: [(0.0, 0, 0), (0.1, 0, 1)])
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {})
    return models, runner.build_protocol(models), source


def test_registration_freezes_development_matrix_and_balanced_policy_orders(registered):
    models, protocol, _ = registered
    assert protocol["lanes"] == [8]
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["arrival_seeds"] == [41, 42, 43]
    assert protocol["arrival_horizon"] == 600.0
    assert protocol["observation_end"] == 1200.0
    assert protocol["budgets"] == {"8": 288}
    assert protocol["reference_budget"] == 4096
    assert protocol["repeats"] == 6
    assert len(protocol["tapes"]) == 24
    assert len(protocol["capacities"]) == len(models) * 2
    assert not protocol["prior_result"]["utility_gate_passed"]
    for tape in protocol["tapes"]:
        orders = protocol["timing_orders"][tape["tape_id"]]
        assert set(map(tuple, orders)) == set(permutations(protocol["labels"]))
        assert all(sum(order[position] == label for order in orders) == 2
                   for label in protocol["labels"] for position in range(3))


@pytest.mark.parametrize("change", ("budget", "arrival", "source", "order", "horizon"))
def test_registered_design_source_or_timing_order_mutation_is_rejected(registered, change):
    models, protocol, source = registered
    if change == "budget":
        protocol["budgets"]["8"] += 1
    elif change == "arrival":
        protocol["tapes"][0]["arrivals"][0][0] += 1.0
    elif change == "source":
        source.write_text("changed selector\n")
    elif change == "order":
        order = protocol["timing_orders"][protocol["tapes"][0]["tape_id"]][0]
        order[0], order[1] = order[1], order[0]
    else:
        protocol["arrival_horizon"] += 1.0
    with pytest.raises(ValueError):
        runner.verify_protocol(protocol, models)


def test_same_tape_is_used_for_all_outputs_and_every_timed_audit_is_deferred(registered, monkeypatch):
    models, protocol, _ = registered
    protocol["tapes"] = protocol["tapes"][:1]
    calls, audit_objects = [], []

    def zero(model, tape, horizon, cutoff, budget, selector, reference_cache, policy):
        calls.append(("zero", policy, tape, budget, selector))
        audit_objects.append(reference_cache)
        return {"policy": policy}

    def measured(model, tape, horizon, cutoff, budget, selector, policy):
        calls.append(("measured", policy, tape, budget, selector))
        return {"policy": policy}, ("selected",)

    def audit(model, result, selections, reference_cache):
        calls.append(("audit", result["policy"]))
        audit_objects.append(reference_cache)
        assert selections == ("selected",)
        return result

    monkeypatch.setattr(legacy, "simulate_trajectory", zero)
    monkeypatch.setattr(replay, "simulate_delayed_trajectory", measured)
    monkeypatch.setattr(replay, "audit_delayed_trajectory", audit)
    row = runner.decision_delay_cell(models[0], 8, protocol, kernels, progress=False)
    assert [call[0] for call in calls] == ["zero"] * 4 + ["measured"] * 18 + ["audit"] * 18
    assert len({id(call[2]) for call in calls[:22]}) == 1
    assert len({id(audit) for audit in audit_objects}) == 1
    assert len({id(call[4]) for call in calls[:4]}) == 4
    assert [call[3] for call in calls[:4]] == [4096, 288, 288, 288]
    assert all(call[3] == 288 for call in calls[4:22])
    assert row["physical_lanes"] == 2
    assert len(row["runs"][0]["measured"]) == 6


@pytest.fixture
def paired_case(registered, monkeypatch):
    models, protocol, _ = registered
    models = models[:1]
    protocol.update(models=[models[0].snapshot()], loads=[0.8], arrival_seeds=[41],
                    execution_order=[[models[0].node, 8]])
    protocol["tapes"] = protocol["tapes"][:1]
    simulate = replay.simulate_delayed_trajectory

    def no_delay(*args, **kwargs):
        return simulate(*args, **kwargs, clock=lambda: 0)

    monkeypatch.setattr(replay, "simulate_delayed_trajectory", no_delay)
    row = runner.decision_delay_cell(models[0], 8, protocol, kernels, progress=False)
    return models, protocol, [row]


def _refresh(result, protocol):
    for cutoff in ("arrival_horizon", "observation_end"):
        result["snapshots"][cutoff] = legacy.trajectory_snapshot(result["jobs"], protocol[cutoff], 3)


def test_identical_correct_outputs_pass_integration_without_claiming_flow_gain(paired_case):
    result = summary.summarize(*paired_case)
    assert result["integration_gate_passed"]
    assert result["utility_gate_passed"]
    assert result["zero_trajectory_identity_passed"]
    assert result["trajectory_count"] == 22
    assert not result["flow_gain_established"]


def test_common_cohort_intersects_all_repeats_and_unfinished_counts_average_within_seed(paired_case):
    models, protocol, rows = deepcopy(paired_case)
    candidate = rows[0]["runs"][0]["measured"][5]["policies"]["integer_capacity"]
    candidate["jobs"][1].update(start=None, completion=None)
    _refresh(candidate, protocol)
    result = summary.summarize(models, protocol, rows)
    snapshot = result["paired_runs"][0]["snapshots"]["observation_end"]
    assert snapshot["common_cohort_count"] == 1
    assert snapshot["measured"]["integer_capacity"]["unfinished_jobs"] == pytest.approx(1 / 6)
    assert snapshot["zero"]["full_reference"]["common_mean_flow_s"] == 2.0
    assert not result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    assert all(group["seed_count"] == 1 and group["repeats_per_seed"] == 6 for group in result["groups"])
    assert all(group["measured"]["integer_capacity"]["unfinished_jobs"] == pytest.approx(1 / 6)
               for group in result["groups"])


def test_measured_trajectory_can_differ_while_zero_delay_identity_remains_required(paired_case):
    models, protocol, rows = deepcopy(paired_case)
    model = replace(models[0], lanes=8)
    times = iter((0, 1_000_000_000, 1_000_000_000, 2_000_000_000))
    candidate, selections = simulate_delayed_trajectory(
        model, protocol["tapes"][0]["arrivals"], protocol["arrival_horizon"],
        protocol["observation_end"], protocol["budgets"]["8"],
        kernels.select_integer_capacity_shared_threshold_frame,
        policy="integer_capacity", clock=lambda: next(times))
    replay.audit_delayed_trajectory(model, candidate, selections, legacy.ReferenceAudit(model))
    rows[0]["runs"][0]["measured"][0]["policies"]["integer_capacity"] = candidate
    result = summary.summarize(models, protocol, rows)
    assert result["integration_gate_passed"]
    assert result["zero_trajectory_identity_passed"]
    rows[0]["runs"][0]["zero"]["integer_capacity"]["jobs"][0]["completion"] += 1.0
    result = summary.summarize(models, protocol, rows)
    assert not result["zero_trajectory_identity_passed"]
    assert not result["integration_gate_passed"]


@pytest.mark.parametrize("failure", ("uncertified", "wrong_action", "incomplete_reference", "cancelled_service"))
def test_bad_measured_certificate_or_execution_cannot_pass_integration(paired_case, failure):
    models, protocol, rows = deepcopy(paired_case)
    candidate = rows[0]["runs"][0]["measured"][0]["policies"]["integer_capacity"]
    if failure == "uncertified":
        candidate["selector_metrics"]["canonical_exact_count"] -= 1
    elif failure == "wrong_action":
        candidate["selector_metrics"]["canonical_action_mismatch_count"] += 1
    elif failure == "incomplete_reference":
        candidate["reference_audit"]["incomplete_frame_count"] += 1
    else:
        candidate["frames"][0]["executed_service"] = [0, 0, 0]
    result = summary.summarize(models, protocol, rows)
    assert not result["integration_gate_passed"]
    assert not result["flow_gain_established"]


def test_completed_only_gain_cannot_hide_a_worse_unfinished_tail(paired_case):
    models, protocol, rows = deepcopy(paired_case)
    for repeat in rows[0]["runs"][0]["measured"]:
        candidate = repeat["policies"]["integer_capacity"]
        candidate["jobs"][0]["completion"] = 1.0
        candidate["jobs"][1].update(start=None, completion=None)
        _refresh(candidate, protocol)
    result = summary.summarize(models, protocol, rows)
    assert all(value > 0.01 for value in result["pooled_flow_improvement"].values())
    assert not result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    for group in result["groups"]:
        for comparison in group["comparisons"].values():
            assert not comparison["checks"]["unfinished_jobs"]
            assert not comparison["checks"]["restricted_mean_flow_s"]
            assert comparison["checks"]["common_mean_flow_s"]
