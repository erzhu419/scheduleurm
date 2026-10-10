"""Catch misleading performance gains and invalid reuse of the serial control."""
from copy import deepcopy
import json

import pytest

from algorithm.experiments.aligned_hardware_performance_necessity import (
    common_metrics, replay_matches, summarize,
)
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot


def test_serial_identity_rejects_changed_dispatch_and_job_completion():
    result = {"jobs": [{"coordinate": 0, "index": 0, "arrival": 0.0, "start": 1.0, "completion": 3.0}],
              "frames": [{"start": 1.0, "queue": (1,), "lane_ids": (0,),
                          "option_indices": (0,), "service": (1,), "duration": 2.0}]}
    old = json.loads(json.dumps(result))
    assert replay_matches(result, old)
    wrong = deepcopy(old)
    wrong["frames"][0]["start"] = 2.0
    assert not replay_matches(result, wrong)
    wrong = deepcopy(old)
    wrong["jobs"][0]["completion"] = 4.0
    assert not replay_matches(result, wrong)


def test_common_completion_cohort_does_not_remove_unfinished_job_cost():
    jobs = [{"coordinate": 0, "index": i, "arrival": 0.0, "start": 0.0, "completion": end}
            for i, end in enumerate((1.0, None))]
    completed = deepcopy(jobs)
    completed[0]["completion"], completed[1]["completion"] = 2.0, 3.0
    policies = {label: {"jobs": population,
                       "snapshots": {"observation_end": trajectory_snapshot(population, 5.0, 3)}}
                for label, population in (("candidate", jobs), ("baseline", completed))}
    stats = common_metrics(policies, 5.0)
    assert stats["candidate"]["common_jobs"] == stats["baseline"]["common_jobs"] == 1
    assert stats["candidate"]["common_mean_flow_s"] < stats["baseline"]["common_mean_flow_s"]
    assert stats["candidate"]["restricted_mean_flow_s"] == 3.0
    assert stats["baseline"]["restricted_mean_flow_s"] == 2.5
    assert stats["candidate"]["unfinished_jobs"] == 1


def _comparison_data():
    labels = ("refill_joint", "refill_single", "enum_maxweight", "delay", "fixed_p1")
    rows = []
    for node in ("a", "b", "c", "d"):
        for load in (.8, .95):
            for seed in (41, 42, 43):
                stats = {label: {"restricted_mean_flow_s": 9.0 if label == "refill_joint" else 10.0,
                                 "common_mean_flow_s": 9.0 if label == "refill_joint" else 10.0,
                                 "common_p90_flow_s": 9.0 if label == "refill_joint" else 10.0,
                                 "unfinished_jobs": 0}
                         for label in labels}
                rows.append({"tape_id": f"{node}:{load}:{seed}", "node": node, "load": load,
                             "snapshots": {"arrival_horizon": deepcopy(stats), "observation_end": stats}})
    protocol = {"experiment": "test", "registered_at_utc": "test", "tapes": [{"tape_id": row["tape_id"]} for row in rows],
                "reused_policies": list(labels[:2]), "capacities": [], "limitations": "test"}
    return rows, protocol


@pytest.mark.parametrize("metric,value", [("common_p90_flow_s", 15.0), ("unfinished_jobs", 1)])
def test_average_gain_cannot_hide_tail_or_unfinished_regression(metric, value):
    rows, protocol = _comparison_data()
    assert summarize(rows, protocol, True, {})["joint_advantage_established"]
    rows[0]["snapshots"]["arrival_horizon"]["refill_joint"][metric] = value
    assert summarize(rows, protocol, True, {})["decision"] == "JOINT_PERFORMANCE_NECESSITY_HOLD"
    assert summarize(rows, protocol, False, {})["decision"] == "REPLAY_HOLD"


def test_missing_registered_tape_is_not_a_complete_performance_result():
    rows, protocol = _comparison_data()
    with pytest.raises(ValueError, match="tapes"):
        summarize(rows[:-1], protocol, True, {})
