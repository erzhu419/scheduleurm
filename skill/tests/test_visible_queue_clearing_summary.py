"""Meaningful flow gains cannot conceal tails, load regressions, or missing pairs."""
from copy import deepcopy

import pytest

from algorithm.experiments.visible_queue_clearing_summary import summarize


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
                        "clearing_rollout": {"restricted_mean_flow_s": 18., "common_mean_flow_s": 18.,
                                             "common_p90_flow_s": 37., "common_jobs": 100,
                                             "unfinished_jobs": 95 if cutoff == "arrival_horizon" else 0},
                    }
                rows.append({**tape, "baseline_reproduction_passed": True, "rollout_checks_passed": True,
                             "snapshots": snapshots,
                             "work": {"maxweight": {"rollout_calls": 0, "maximum_rollout_calls": 0},
                                      "clearing_rollout": {"candidate_count": 3, "rollout_calls": 11,
                                                           "maximum_rollout_calls": seed - 40}}})
    return rows, protocol


def test_gain_allows_small_paired_regression_and_preserves_peak_not_sum():
    rows, protocol = _case()
    rows[0]["snapshots"]["observation_end"]["clearing_rollout"]["restricted_mean_flow_s"] = 22.
    result = summarize(rows, protocol)
    assert result["decision"] == "VISIBLE_QUEUE_CLEARING_DEVELOPMENT_ADVANCE"
    assert len(result["groups"]) == 16 and len(result["loads"]) == 2
    assert result["independent_seed_replicates"] == 3
    assert all(seed["paired_cells"] == 8 for seed in result["seed_aggregates"])
    assert result["totals_work"]["clearing_rollout"] == {
        "candidate_count": 72, "rollout_calls": 264, "maximum_rollout_calls": 3}
    assert "restricted_mean_flow_s" in result["paired_runs"][0]["comparisons"]["observation_end"]["regressed_metrics"]


def test_mean_gain_cannot_hide_pooled_unfinished_increase_within_group_tolerance():
    rows, protocol = _case()
    for row in rows:
        row["snapshots"]["arrival_horizon"]["clearing_rollout"]["unfinished_jobs"] = 101
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"] and result["group_secondary_passed"]
    assert not result["pooled_unfinished_checks"]["arrival_horizon"]
    assert result["decision"] == "VISIBLE_QUEUE_CLEARING_PERFORMANCE_HOLD"


def test_zero_baseline_unfinished_must_stay_zero():
    rows, protocol = _case()
    rows[0]["snapshots"]["observation_end"]["clearing_rollout"]["unfinished_jobs"] = 1
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"]
    assert result["failed_secondary_groups"][0]["failed_metrics"] == ["unfinished_jobs"]
    assert not result["performance_gain_established"]


def test_group_p90_regression_is_retained_even_when_pooled_mean_improves():
    rows, protocol = _case()
    for row in rows[:3]:
        row["snapshots"]["observation_end"]["clearing_rollout"]["common_p90_flow_s"] = 42.1
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"]
    assert result["failed_secondary_groups"][0]["failed_metrics"] == ["common_p90_flow_s"]
    assert result["decision"] == "VISIBLE_QUEUE_CLEARING_PERFORMANCE_HOLD"


def test_one_load_mean_regression_cannot_be_hidden_by_gain_at_the_other():
    rows, protocol = _case()
    for row in rows:
        row["snapshots"]["observation_end"]["clearing_rollout"]["restricted_mean_flow_s"] = 21. if row["load"] == .95 else 15.
    result = summarize(rows, protocol)
    assert result["flow_gain_threshold_passed"] and not result["load_mean_nonregression_passed"]
    assert result["loads"][1]["comparisons"]["observation_end"]["deltas"]["restricted_mean_flow_s"] == 1.
    assert result["decision"] == "VISIBLE_QUEUE_CLEARING_PERFORMANCE_HOLD"


def test_execution_failure_overrides_performance_gain():
    rows, protocol = _case()
    for field in ("baseline_reproduction_passed", "rollout_checks_passed"):
        changed = deepcopy(rows)
        changed[0][field] = False
        result = summarize(changed, protocol)
        assert result["flow_gain_threshold_passed"] and not result["execution_checks_passed"]
        assert result["decision"] == "VISIBLE_QUEUE_CLEARING_REPLAY_HOLD"


def test_missing_tape_and_mispaired_identity_are_rejected():
    rows, protocol = _case()
    with pytest.raises(ValueError, match="registered"):
        summarize(rows[:-1], protocol)
    rows[0]["tape_id"], rows[1]["tape_id"] = rows[1]["tape_id"], rows[0]["tape_id"]
    with pytest.raises(ValueError, match="identity"):
        summarize(rows, protocol)
