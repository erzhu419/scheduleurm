"""A valid rollout ledger cannot hide tail-constraint violations or age reads."""
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_age_tail_constrained_lane_rollout import rollout_checks, work_totals


def _frame():
    return dict(start=5., queue=(2,), waiting_ages=((10., 1.),), rollout=dict(
        queue_limit=4, eligible=True, candidate_count=2, rollout_calls=10,
        rollout_option_probes=20, root_option_probes=4, window_s=2., sample_count=8,
        planning_overflow=False, changed=True, baseline_predicted_completion_s=20.,
        chosen_predicted_completion_s=19., baseline_predicted_squared_flow_s2=100.,
        chosen_predicted_squared_flow_s2=99., risk_feasible_candidates=1,
        risk_filtered_candidates=1, flow_evaluations=20, waiting_age_reads=2,
        risk_changed_unconstrained_root=True, planning_events=3, planning_random_draws=5,
        planning_rate_reads=1, rollout_idle_calls=4))


def test_predicted_second_moment_violation_and_wrong_fifo_age_order_fail():
    model = SimpleNamespace(lanes=2, options=[SimpleNamespace(requirements=(1,)),
                                             SimpleNamespace(requirements=(2,))])
    protocol = dict(planning_samples=8, planning_event_cap=128)
    frame = _frame()
    assert rollout_checks(dict(frames=[frame]), model, protocol)
    frame["rollout"]["chosen_predicted_squared_flow_s2"] = 101.
    assert not rollout_checks(dict(frames=[frame]), model, protocol)
    frame["rollout"]["chosen_predicted_squared_flow_s2"] = 99.
    frame["waiting_ages"] = ((1., 10.),)
    assert not rollout_checks(dict(frames=[frame]), model, protocol)


def test_new_preprocessing_scans_and_all_context_age_reads_are_in_work_total():
    planner = SimpleNamespace(preprocessing_work=dict(option_probes=2, age_profile_option_probes=2,
                                                       completion_offset_reads=3))
    result = dict(frames=[_frame()], rollout_work=dict(root_option_probes=4, rollout_option_probes=20,
        rollout_calls=10, maximum_rollout_calls=10, age_context_job_reads=200))
    total = work_totals(result, planner, 10.)
    assert total["total_option_probes"] == 28
    assert total["age_context_job_reads"] == 200
    assert total["waiting_age_reads"] == 2 and total["flow_evaluations"] == 20
    assert total["risk_changed_unconstrained_calls"] == 1 and total["max_rollout_calls"] == 10
