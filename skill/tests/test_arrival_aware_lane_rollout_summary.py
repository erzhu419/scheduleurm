"""Flow gains must survive the frozen load, tail and replay criteria."""
from copy import deepcopy

import pytest

from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize


def _case():
    models = [{"node": f"node{i}", "lanes": 2 if i < 2 else 4} for i in range(4)]
    protocol = {"experiment": "toy", "registered_at_utc": "frozen", "limitations": "development",
                "models": models, "loads": [.8, .95], "arrival_seeds": [41, 42, 43],
                "performance": {"min_gain": .01, "group_secondary_tolerance": .05}, "tapes": []}
    rows = []
    for model in models:
        for load in protocol["loads"]:
            for seed in protocol["arrival_seeds"]:
                tape = {**model, "load": load, "seed": seed,
                        "tape_id": f"{model['node']}:{load}:{seed}"}
                protocol["tapes"].append(tape)
                snapshots = {}
                for cutoff in ("arrival_horizon", "observation_end"):
                    snapshots[cutoff] = {
                        "maxweight": {"restricted_mean_flow_s": 20., "common_mean_flow_s": 20.,
                                      "common_p90_flow_s": 40., "common_jobs": 100,
                                      "unfinished_jobs": 100 if cutoff == "arrival_horizon" else 0},
                        "arrival_rollout": {"restricted_mean_flow_s": 18., "common_mean_flow_s": 18.,
                                            "common_p90_flow_s": 37., "common_jobs": 100,
                                            "unfinished_jobs": 95 if cutoff == "arrival_horizon" else 0},
                    }
                rows.append({**tape, "baseline_reproduction_passed": True, "rollout_checks_passed": True,
                             "snapshots": snapshots,
                             "work": {"maxweight": {"rollout_calls": 0, "max_rollout_calls": 0},
                                      "arrival_rollout": {"planning_events": 8, "planning_random_draws": 16,
                                                          "planning_overflows": 0, "rollout_calls": 11,
                                                          "max_rollout_calls": seed - 40}}})
    return rows, protocol


def test_pass_preserves_paired_seed_scope_and_work_peak():
    rows, protocol = _case()
    rows[0]["snapshots"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"] = 22.
    result = summarize(rows, protocol)
    assert result["decision"] == "ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_PASS"
    assert len(result["groups"]) == 16 and len(result["loads"]) == 2
    assert result["tapes"] == 24 and result["trajectories"] == 48
    assert result["independent_seed_replicates"] == 3
    assert all(seed["paired_cells"] == 8 for seed in result["seed_aggregates"])
    assert result["totals_work"]["arrival_rollout"] == {
        "planning_events": 192, "planning_random_draws": 384,
        "planning_overflows": 0, "rollout_calls": 264, "max_rollout_calls": 3}
    assert "restricted_mean_flow_s" in result["paired_runs"][0]["comparisons"]["observation_end"]["regressed_metrics"]


@pytest.mark.parametrize("flow,passed", [(19.8, True), (19.800001, False), (20., False)])
def test_exact_one_percent_primary_boundary(flow, passed):
    rows, protocol = _case()
    for row in rows:
        row["snapshots"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"] = flow
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"] is passed
    assert result["performance_gain_established"] is passed


def test_one_load_regression_cannot_be_hidden_by_another_load_gain():
    rows, protocol = _case()
    for row in rows:
        row["snapshots"]["observation_end"]["arrival_rollout"]["restricted_mean_flow_s"] = 21. if row["load"] == .95 else 15.
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"] and not result["load_mean_nonregression_passed"]
    assert result["decision"] == "ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD"


@pytest.mark.parametrize("failure", ["p90", "zero_unfinished", "pooled_unfinished"])
def test_mean_gain_cannot_hide_group_or_pooled_tail_regression(failure):
    rows, protocol = _case()
    if failure == "p90":
        for row in rows[:3]:
            row["snapshots"]["observation_end"]["arrival_rollout"]["common_p90_flow_s"] = 42.1
    elif failure == "zero_unfinished":
        rows[0]["snapshots"]["observation_end"]["arrival_rollout"]["unfinished_jobs"] = 1
    else:
        for row in rows:
            row["snapshots"]["arrival_horizon"]["arrival_rollout"]["unfinished_jobs"] = 101
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"] and result["execution_checks_passed"]
    assert result["decision"] == "ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD"
    if failure == "pooled_unfinished":
        assert result["group_secondary_passed"]
        assert not result["pooled_unfinished_checks"]["arrival_horizon"]
    else:
        expected_metric = "common_p90_flow_s" if failure == "p90" else "unfinished_jobs"
        assert result["failed_secondary_groups"][0]["failed_metrics"] == [expected_metric]


@pytest.mark.parametrize("field", ["baseline_reproduction_passed", "rollout_checks_passed"])
def test_operational_failure_is_distinct_from_performance_hold(field):
    rows, protocol = _case()
    rows[0][field] = False
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"]
    assert not result["operational_integrity_passed"]
    assert not result["performance_gain_established"]
    assert result["decision"] == "ARRIVAL_AWARE_LANE_ROLLOUT_REPLAY_FAIL"


def test_missing_tape_and_mispaired_identity_are_rejected():
    rows, protocol = _case()
    with pytest.raises(ValueError, match="registered"):
        summarize(rows[:-1], protocol)
    changed = deepcopy(rows)
    changed[0]["tape_id"], changed[1]["tape_id"] = changed[1]["tape_id"], changed[0]["tape_id"]
    with pytest.raises(ValueError, match="identity"):
        summarize(changed, protocol)
