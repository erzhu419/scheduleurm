"""Check frozen paired inputs and trajectory decision gates with tiny records."""
from copy import deepcopy
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_trajectory as runner
from algorithm.experiments import integer_capacity_trajectory_summary as summary
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


def _model(node="tiny", duration=2.0):
    return FrameModel(
        node=node, scenario_id="paired_test", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (1, 0, 0), duration, ((duration,), (), ())),),
    )


@pytest.fixture
def registered(tmp_path, monkeypatch):
    models = tuple(_model(f"node{i}", float(i + 1)) for i in range(4))
    timing = tmp_path / "timing_protocol.json"
    source = tmp_path / "selector.py"
    source.write_text("frozen selector\n")
    timing.write_text(json.dumps({
        "models": [model.snapshot() for model in models],
        "execution_sources": {"selector.py": source.read_text()},
        "search_counters": ["lp_calls"], "capacity_counters": ["region_visits"],
    }))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "TIMING_PROTOCOL", timing)
    monkeypatch.setattr(runner, "TIMING_OUTPUT", tmp_path / "timing.json")
    monkeypatch.setattr(runner, "EXTRA_SOURCES", ())
    monkeypatch.setattr(runner, "require_timing_advance", lambda: None)
    calls = []

    def capacity(model, load):
        calls.append((model.node, model.lanes, load))
        return {"arrival_rates": [load * model.lanes / model.options[0].duration, 0, 0]}

    def tape(rates, horizon, seed):
        assert horizon == 600.0
        return [(0.01 * rates[0], 0, seed)]

    monkeypatch.setattr(runner, "capacity_certificate", capacity)
    monkeypatch.setattr(runner, "poisson_arrivals", tape)
    return models, runner.build_protocol(models), calls, source


def test_registration_uses_each_model_and_virtual_lane_capacity(registered):
    models, protocol, calls, _ = registered
    assert protocol["lanes"] == [4, 8, 16]
    assert protocol["loads"] == [0.8, 0.95]
    assert protocol["arrival_seeds"] == list(range(2026092711, 2026092715))
    assert protocol["arrival_horizon"] == 600.0
    assert protocol["observation_end"] == 1200.0
    assert protocol["budgets"] == {"4": 144, "8": 288, "16": 576}
    assert protocol["reference_budget"] == 4096
    expected = {(model.node, lanes, load) for model in models
                for lanes in (4, 8, 16) for load in (0.8, 0.95)}
    assert set(calls) == expected
    assert len(calls) == 24
    assert len(protocol["tapes"]) == 96
    assert len({tape["tape_id"] for tape in protocol["tapes"]}) == 96
    for tape in protocol["tapes"]:
        model = next(model for model in models if model.node == tape["node"])
        assert tape["arrivals"][0][0] == pytest.approx(
            0.01 * tape["load"] * tape["lanes"] / model.options[0].duration)


@pytest.mark.parametrize("change", ("budget", "arrival", "source", "horizon"))
def test_changed_registered_design_or_source_is_rejected(registered, change):
    models, protocol, _, source = registered
    if change == "budget":
        protocol["budgets"]["4"] += 1
    elif change == "arrival":
        protocol["tapes"][0]["arrivals"][0][0] += 1.0
    elif change == "source":
        source.write_text("changed selector\n")
    else:
        protocol["arrival_horizon"] += 1.0
    with pytest.raises(ValueError):
        runner.verify_protocol(protocol, models)


def test_all_policies_receive_identical_tape_and_only_audit_cache_is_shared(registered, monkeypatch):
    models, protocol, _, _ = registered
    model = models[0]
    protocol["tapes"] = [protocol["tapes"][0]]
    calls = []

    def simulate(model, tape, horizon, cutoff, budget, selector, reference_cache, policy):
        calls.append((model, tape, horizon, cutoff, budget, selector, reference_cache, policy))
        return {"policy": policy}

    monkeypatch.setattr(runner, "simulate_trajectory", simulate)
    row = runner.trajectory_cell(model, 4, protocol, progress=False)
    assert len(calls) == 4
    assert len({id(call[1]) for call in calls}) == 1
    assert len({id(call[6]) for call in calls}) == 1
    assert len({id(call[5]) for call in calls}) == 4
    assert [call[4] for call in calls] == [4096, 144, 144, 144]
    assert all(call[0].node == model.node and call[0].lanes == 4 for call in calls)
    assert row["physical_lanes"] == 2
    assert len(row["runs"]) == 1


def test_common_completed_cohort_intersects_actual_coordinate_and_job_ids():
    def jobs(completions):
        return [
            {"coordinate": coordinate, "index": index, "arrival": 0.0,
             "start": 0.0, "completion": completion}
            for (coordinate, index), completion in zip(((0, 0), (0, 1), (0, 2), (1, 0)), completions)
        ]

    policies = {
        "full_reference": {"jobs": jobs((2.0, None, 8.0, 1.0))},
        "old_shared": {"jobs": jobs((4.0, 3.0, None, 1.0))},
        "fractional_shared": {"jobs": jobs((3.0, 6.0, 9.0, 1.0))},
        "integer_capacity": {"jobs": jobs((2.0, None, 8.0, 1.0))},
    }
    cohort = summary.common_cohort(policies, 5.0)
    assert cohort["job_ids"] == [[0, 0], [1, 0]]
    assert cohort["count"] == 2
    assert cohort["policies"]["old_shared"]["mean_flow_s"] == 2.5
    assert cohort["policies"]["old_shared"]["p90_flow_s"] == pytest.approx(3.7)
    assert summary.common_cohort(policies, 3.0)["job_ids"] == [[1, 0]]


@pytest.fixture
def paired_case(registered):
    models, protocol, _, _ = registered
    models = models[:1]
    protocol.update(models=[models[0].snapshot()], lanes=[4], loads=[0.8],
                    arrival_seeds=[2026092711], execution_order=[[models[0].node, 4]])
    protocol["tapes"] = [protocol["tapes"][0]]
    protocol["tapes"][0]["arrivals"] = [[0.0, 0, 0], [0.1, 0, 1]]
    row = runner.trajectory_cell(models[0], 4, protocol, progress=False)
    return models, protocol, [row]


def test_identical_correct_trajectories_pass_preservation_without_claiming_flow_gain(paired_case):
    result = summary.summarize(*paired_case)
    assert result["integration_gate_passed"]
    assert result["utility_gate_passed"]
    assert result["trajectory_identity_passed"]
    assert not result["flow_gain_established"]


@pytest.mark.parametrize("failure", ("uncertified", "wrong_score", "wrong_action", "incomplete_reference"))
def test_candidate_incomplete_or_incorrect_cannot_pass_integration(paired_case, failure):
    models, protocol, rows = deepcopy(paired_case)
    candidate = rows[0]["runs"][0]["policies"]["integer_capacity"]
    if failure == "uncertified":
        candidate["selector_metrics"]["canonical_exact_count"] -= 1
    elif failure == "wrong_score":
        candidate["selector_metrics"]["score_mismatch_count"] += 1
    elif failure == "wrong_action":
        candidate["selector_metrics"]["canonical_action_mismatch_count"] += 1
    else:
        candidate["reference_audit"]["incomplete_count"] += 1
        candidate["reference_audit"]["incomplete_frame_count"] += 1
    result = summary.summarize(models, protocol, rows)
    assert not result["integration_gate_passed"]
    assert not result["flow_gain_established"]


def test_lower_completed_mean_cannot_hide_worse_restricted_flow_and_unfinished_tail(paired_case):
    models, protocol, rows = deepcopy(paired_case)
    policies = rows[0]["runs"][0]["policies"]
    candidate = policies["integer_capacity"]
    candidate["jobs"][1].update(start=None, completion=None)
    for label in ("arrival_horizon", "observation_end"):
        candidate["snapshots"][label] = trajectory_snapshot(candidate["jobs"], protocol[label], 3)
    assert candidate["snapshots"]["observation_end"]["completed_mean_flow_s"] < (
        policies["old_shared"]["snapshots"]["observation_end"]["completed_mean_flow_s"])
    result = summary.summarize(models, protocol, rows)
    assert not result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    for group in result["groups"]:
        for comparison in group["comparisons"].values():
            assert not comparison["checks"]["unfinished_jobs"]
            assert not comparison["checks"]["restricted_mean_flow_s"]
            assert comparison["checks"]["common_mean_flow_s"]


def test_empty_common_cohort_cannot_establish_utility_nonregression(paired_case):
    models, protocol, _ = deepcopy(paired_case)
    protocol.update(arrival_horizon=0.5, observation_end=0.75)
    row = runner.trajectory_cell(models[0], 4, protocol, progress=False)
    result = summary.summarize(models, protocol, [row])
    assert result["integration_gate_passed"]
    assert not result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    assert all(pair["snapshots"][cutoff]["common_cohort"]["count"] == 0
               for pair in result["paired_runs"] for cutoff in ("arrival_horizon", "observation_end"))
