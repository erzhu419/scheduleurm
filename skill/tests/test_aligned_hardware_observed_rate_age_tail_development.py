"""Birth exposure errors and missing estimator work must block replay claims."""
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import rollout_checks, work_totals


def _frame():
    return dict(start=20., queue=(2,), waiting_ages=((10., 1.),), observed_arrival_counts=(7,), rollout=dict(
        queue_limit=4, eligible=True, candidate_count=2, rollout_calls=10,
        rollout_option_probes=20, root_option_probes=4, window_s=2., sample_count=8,
        planning_overflow=False, changed=True, baseline_predicted_completion_s=20.,
        chosen_predicted_completion_s=19., baseline_predicted_squared_flow_s2=100.,
        chosen_predicted_squared_flow_s2=99., risk_feasible_candidates=1,
        risk_filtered_candidates=1, flow_evaluations=20, waiting_age_reads=2,
        risk_changed_unconstrained_root=True, planning_events=3, planning_random_draws=5,
        planning_rate_reads=1, rollout_idle_calls=4, observed_arrival_counts=(7,),
        estimated_arrival_rates=(.35,), rate_exposure_s=20., rate_estimation_coordinate_reads=1,
        rate_estimation_divisions=1))


def test_estimate_cannot_drop_already_admitted_births_or_decay_after_h():
    model = SimpleNamespace(lanes=2, workloads=("a",), options=[SimpleNamespace(requirements=(1,)),
                                                              SimpleNamespace(requirements=(2,))])
    protocol = dict(planning_samples=8, planning_event_cap=128, arrival_horizon=600.)
    frame = _frame()
    assert rollout_checks(dict(frames=[frame]), model, protocol)
    frame["rollout"]["estimated_arrival_rates"] = (.1,)  # Q/t loses 5 admitted births.
    assert not rollout_checks(dict(frames=[frame]), model, protocol)
    frame["start"] = 1200.
    frame["rollout"].update(rate_exposure_s=600., estimated_arrival_rates=(7 / 600.,))
    assert rollout_checks(dict(frames=[frame]), model, protocol)
    frame["rollout"].update(rate_exposure_s=1200., estimated_arrival_rates=(7 / 1200.,))
    assert not rollout_checks(dict(frames=[frame]), model, protocol)


def test_dimension_probe_rate_estimation_and_causal_engine_costs_are_charged():
    planner = SimpleNamespace(preprocessing_work=dict(option_probes=2, age_profile_option_probes=2,
                                                       rate_dimension_option_probes=1))
    result = dict(frames=[_frame()], rollout_work=dict(root_option_probes=4, rollout_option_probes=20,
        rollout_calls=10, maximum_rollout_calls=10, observed_arrival_counter_updates=7,
        observed_arrival_context_coordinate_reads=1))
    total = work_totals(result, planner, 600.)
    assert total["total_option_probes"] == 29
    assert total["observed_arrival_counter_updates"] == 7
    assert total["observed_arrival_context_coordinate_reads"] == 1
    assert total["rate_estimation_coordinate_reads"] == total["rate_estimation_divisions"] == 1
