"""The replay ledger must include planning and preprocessing work."""
from types import SimpleNamespace

from algorithm.experiments.aligned_hardware_arrival_aware_lane_rollout import rollout_checks, work_totals


def _frame(now, **updates):
    fields = dict(queue_limit=4, eligible=True, candidate_count=1, rollout_calls=10,
                  rollout_option_probes=20, root_option_probes=4, window_s=2.,
                  sample_count=8, planning_overflow=False, changed=True,
                  baseline_predicted_completion_s=20., chosen_predicted_completion_s=19.,
                  planning_events=3, planning_random_draws=5, planning_rate_reads=2,
                  rollout_idle_calls=4)
    fields.update(updates)
    return dict(start=now, queue=(2,), rollout=fields)


def _model():
    return SimpleNamespace(lanes=2, options=[SimpleNamespace(requirements=(1,)),
                                            SimpleNamespace(requirements=(2,))])


def _protocol():
    return dict(planning_samples=8, planning_event_cap=128)


def test_generated_arrivals_and_constructor_scans_are_charged_once():
    planner = SimpleNamespace(preprocessing_work=dict(option_probes=2, requirement_reads=2,
                                                       completion_offset_reads=3, singleton_checks=2))
    result = dict(frames=[_frame(5.), _frame(10.)], rollout_work=dict(
        root_option_probes=8, rollout_option_probes=40, rollout_calls=20,
        maximum_rollout_calls=10, eligible_decisions=2, changed_decisions=2, candidate_count=2))
    total = work_totals(result, planner, 10.)
    assert total["total_option_probes"] == 50
    assert total["preprocessing_option_probes"] == 2
    assert total["planning_events"] == 6 and total["planning_random_draws"] == 10
    assert total["planning_rate_reads"] == 4 and total["rollout_idle_calls"] == 8
    assert total["pre_h_changed_calls"] == total["post_h_changed_calls"] == 1
    assert total["max_rollout_calls"] == 10


def test_extra_continuation_work_and_predicted_regression_block_interpretation():
    assert rollout_checks(dict(frames=[_frame(0.)]), _model(), _protocol())
    cap = 8 * (2 + 128 - 1 + 2 * (2 + 1))
    too_many = _frame(0., rollout_calls=cap + 1, rollout_option_probes=2 * (cap + 1))
    assert not rollout_checks(dict(frames=[too_many]), _model(), _protocol())
    worse = _frame(0., chosen_predicted_completion_s=21.)
    assert not rollout_checks(dict(frames=[worse]), _model(), _protocol())


def test_overflow_uses_only_the_baseline_without_truncated_scoring():
    fallback = _frame(0., planning_overflow=True, changed=False, candidate_count=0,
                      rollout_calls=0, rollout_option_probes=0, root_option_probes=2)
    assert rollout_checks(dict(frames=[fallback]), _model(), _protocol())
    fallback["rollout"]["candidate_count"] = 1
    assert not rollout_checks(dict(frames=[fallback]), _model(), _protocol())
