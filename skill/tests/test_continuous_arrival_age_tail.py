import inspect

import pytest

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import (
    build_protocol, cohort_metrics, queue_area, SEEDS,
)


OPTIONS = (FrameOption("a1", (1, 0), 2., ((2.,), ())),
           FrameOption("a2", (2, 0), 3., ((3., 3.), ())),
           FrameOption("b1", (0, 1), 1., ((), (1.,))))


def test_no_stop_parameter_and_planning_continues_beyond_old_h():
    assert "arrival_horizon" not in inspect.signature(ContinuousArrivalAgeTailLaneRollout).parameters
    planner = ContinuousArrivalAgeTailLaneRollout(OPTIONS, "continuous-test")
    decision = planner(OPTIONS, (2, 1), (1500., 1503.), 1500., 0,
                       ((10., 10.), (0.,)), (150, 75))
    assert decision.rate_exposure_s == 1500.
    assert decision.estimated_arrival_rates == (.1, .05)
    assert decision.window_s == 3.
    assert decision.planning_random_draws > 0
    assert decision.sample_count == 8
    with pytest.raises(TypeError):
        ContinuousArrivalAgeTailLaneRollout(OPTIONS, "test", arrival_horizon=600.)


def test_future_tape_cannot_change_past_decisions():
    model = FrameModel("test", "continuous", 2, ("a", "b"), (1., 1.), OPTIONS)
    prefix = [(0., 0, 0), (0., 0, 1), (0., 1, 0)]
    results = [simulate_observed_rate_contextual_trajectory(model, prefix + future, 60., 60.,
               ContinuousArrivalAgeTailLaneRollout(OPTIONS, "shared"), policy="candidate")
               for future in ([(10., 0, 2)], [(20., 1, 1)])]
    assert [f for f in results[0]["frames"] if f["start"] < 10.] == [
        f for f in results[1]["frames"] if f["start"] < 10.]


def test_cohort_keeps_censored_jobs_and_excludes_outside_births():
    jobs = [{"coordinate": 0, "index": i, "arrival": arrival,
             "start": arrival, "completion": completion}
            for i, (arrival, completion) in enumerate([(299., 1201.), (300., 310.),
                                                       (600., 1201.), (901., 902.)])]
    control = {"jobs": jobs, "snapshots": {"observation_end": {"per_coordinate": [{}] * 3}}}
    candidate = {**control, "jobs": [{**job, "completion": 1100. if job["index"] == 2
                                      else job["completion"]} for job in jobs]}
    metrics = cohort_metrics({"maxweight": control, "arrival_rollout": candidate}, 1200., [300., 900.])
    assert metrics["maxweight"]["restricted_mean_flow_s"] == 305.
    assert metrics["arrival_rollout"]["restricted_mean_flow_s"] == 255.
    assert metrics["maxweight"]["unfinished_jobs"] == 1
    assert metrics["arrival_rollout"]["unfinished_jobs"] == 0
    assert metrics["maxweight"]["common_jobs"] == 1
    assert metrics["maxweight"]["common_p90_flow_s"] == 10.


def test_queue_area_includes_backlog_born_before_middle_cohort():
    result = {"jobs": [{"arrival": 250., "start": 350.},
                       {"arrival": 800., "start": None},
                       {"arrival": 950., "start": None}]}
    assert queue_area(result) == 150.


def test_frozen_development_scope_and_new_confirmation_seeds():
    protocol = build_protocol("development")
    assert len(protocol["tapes"]) == 24
    assert protocol["birth_cohort"] == [300., 900.]
    assert protocol["environment_arrival_end"] == protocol["observation_end"] == 1200.
    assert all(any(arrival[0] > 900. for arrival in tape["arrivals"]) for tape in protocol["tapes"])
    assert SEEDS["confirmation"] == tuple(range(100001, 100013))
