"""A one-action intervention keeps the true prefix and then uses ordinary MW."""
from copy import deepcopy
import json

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.contextual_lane_refill_replay import simulate_contextual_trajectory
from algorithm.experiments import visible_queue_first_change_diagnosis as diagnosis
from algorithm.visible_queue_clearing_rollout import select_context_maxweight, select_visible_queue_clearing


def _case(change=True):
    model = FrameModel("toy", "toy", 2, ("a", "b", "c"), (1., 1., 1.), (
        FrameOption("a", (1, 0, 0), 2., ((2.,), (), ())),
        FrameOption("b", (0, 1, 0), 1., ((), (1.,), ())),
        FrameOption("c", (0, 0, 1), 10., ((), (), (10.,))),
        FrameOption("c_batch", (0, 0, 3), 100., ((), (), (100.,) * 3)),
    ))
    arrivals = [(0., 2, 0), *[(.1, 0, index) for index in range(5)]]
    if change:
        arrivals.append((.1, 1, 0))
    tape = {"node": "toy", "lanes": 2, "load": .8, "seed": 41, "tape_id": "toy:41", "arrivals": arrivals}
    protocol = {"arrival_horizon": 15., "observation_end": 30., "diagnostic_zero_tolerance_s": 1e-9,
                "experiment": "toy", "registered_at_utc": "frozen", "arrival_seeds": [41], "limitations": "toy"}
    reused = {label: simulate_contextual_trajectory(model, arrivals, 2., 30., selector, policy=label)
              for label, selector in (("maxweight", select_context_maxweight), ("clearing_rollout", select_visible_queue_clearing))}
    return model, tape, protocol, json.loads(json.dumps(reused))


def test_first_changed_root_matches_full_policy_and_later_calls_are_plain_maxweight():
    model, tape, protocol, reused = _case()
    selector = diagnosis.FirstChangeMaxWeight()
    result = simulate_contextual_trajectory(model, tape["arrivals"], 2., 30., selector, policy=diagnosis.ONE_SHOT)
    assert selector.first_change_frame == 2
    assert diagnosis.prefix_matches(result, reused["clearing_rollout"], reused["maxweight"], 2)
    assert sum(frame["rollout"]["changed"] for frame in result["frames"]) == 1
    for frame in result["frames"][3:]:
        expected = select_context_maxweight(model.options, frame["queue"], frame["available_before"], frame["start"], frame["lane_ids"][0])
        assert frame["option_indices"] == expected.selection.option_indices
        assert frame["rollout"]["rollout_calls"] == 0


def test_no_changed_root_reproduces_baseline_and_remains_in_diagnostic():
    row = diagnosis.run_tape(*_case(change=False))
    assert row["prefix_and_first_action_match"] and row["first_change"] is None
    effect = row["job_time_effects_vs_maxweight"]["observation_end"][diagnosis.ONE_SHOT]
    assert effect["restricted_mean_delta_s"] == 0
    assert effect["restricted_flow"]["earlier"] == effect["restricted_flow"]["later"] == 0


def test_prefix_available_commitment_or_first_root_mismatch_is_rejected():
    model, tape, protocol, reused = _case()
    selector = diagnosis.FirstChangeMaxWeight()
    result = simulate_contextual_trajectory(model, tape["arrivals"], 2., 30., selector, policy=diagnosis.ONE_SHOT)
    for frame_index, field, value in ((0, "available_before", [20., 0.]), (2, "option_indices", [0])):
        changed = deepcopy(result)
        changed["frames"][frame_index][field] = value
        assert not diagnosis.prefix_matches(changed, reused["clearing_rollout"], reused["maxweight"], 2)


def test_restricted_job_effect_retains_unfinished_tail_and_changed_completion():
    baseline = {"jobs": [{"coordinate": 0, "index": 0, "arrival": 0., "start": 1., "completion": 2.},
                         {"coordinate": 0, "index": 1, "arrival": 1., "start": 2., "completion": 4.}]}
    first = deepcopy(baseline)
    first["jobs"][1]["start"] = first["jobs"][1]["completion"] = None
    effect = diagnosis.job_effects(first, baseline, 10., 1e-9)
    assert effect["restricted_flow"]["total_delta_s"] == 6.
    assert effect["restricted_mean_delta_s"] == 3.
    assert effect["restricted_flow"]["later"] == 1
    assert effect["completion"]["compared_jobs"] == 1


def test_diagnostic_reports_realized_gain_or_loss_without_promoting_predicted_cost():
    model, tape, protocol, reused = _case()
    row = diagnosis.run_tape(model, tape, protocol, reused)
    evidence = row["first_change"]
    assert row["prefix_and_first_action_match"] and evidence["predicted_gain_s"] > 0
    assert evidence["old_root"]["action_id"] == "a" and evidence["new_root"]["action_id"] == "b"
    assert evidence["busy_remaining_s"] == [9., 0.]
    result = diagnosis.summarize([row], protocol)
    assert result["decision"] == "FIRST_CHANGE_DIAGNOSIS_COMPLETE"
    assert result["first_change_tapes"] == 1
    assert result["improved_tapes"] + result["worse_tapes"] + result["unchanged_tapes"] == 1
    row["prefix_and_first_action_match"] = False
    assert diagnosis.summarize([row], protocol)["decision"] == "FIRST_CHANGE_DIAGNOSIS_REPLAY_HOLD"


def test_visible_ids_exclude_jobs_admitted_by_an_earlier_call_at_the_same_time():
    baseline = {"jobs": [{"coordinate": 0, "index": 0, "arrival": 0., "start": 0., "completion": 2.},
                         {"coordinate": 0, "index": 1, "arrival": 0., "start": 1., "completion": 3.},
                         {"coordinate": 0, "index": 2, "arrival": .5, "start": 2., "completion": 4.}]}
    intervention = deepcopy(baseline)
    intervention["jobs"][1]["start"], intervention["jobs"][1]["completion"] = 0., 1.
    intervention["jobs"][2]["start"], intervention["jobs"][2]["completion"] = 3., 5.
    intervention["frames"] = [{"start": 0., "service": (1, 0, 0)},
                              {"start": 0., "queue": (1, 0, 0), "service": (1, 0, 0)}]
    protocol = {"arrival_horizon": 1., "observation_end": 10., "diagnostic_zero_tolerance_s": 1e-9}
    groups = diagnosis.first_change_job_groups(intervention, baseline, 1, protocol)
    assert groups["job_ids"] == {"visible_waiting": [(0, 1)], "previously_admitted": [(0, 0)], "future_arrivals": [(0, 2)]}
    effects = groups["snapshots"]["observation_end"]
    assert effects["visible_waiting"]["restricted_flow"]["total_delta_s"] == -2.
    assert effects["future_arrivals"]["restricted_flow"]["total_delta_s"] == 1.
    assert effects["previously_admitted"]["restricted_flow"]["total_delta_s"] == 0.
