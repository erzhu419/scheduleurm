"""Exercise causal decisions and finite cutoffs with a deterministic solve clock."""
from dataclasses import replace

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import (
    select_canonical_count_space_bnb_frame,
)
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments import integer_capacity_decision_delay_replay as replay
from algorithm.experiments.integer_capacity_trajectory_replay import (
    ReferenceAudit, simulate_trajectory,
)


class SolveClock:
    """Return two timestamps per selector call, with known elapsed seconds."""

    def __init__(self, *delays):
        self.delays = iter(delays)
        self.calls = 0
        self.now = 0

    def __call__(self):
        if self.calls % 2:
            self.now += round(next(self.delays) * 1_000_000_000)
        self.calls += 1
        return self.now


def _model(*, duration=2.0, offsets=(2.0,), lanes=1):
    return FrameModel(
        node="small_delay", scenario_id="finite_decision_cutoff", lanes=lanes,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (len(offsets), 0, 0), duration, (offsets, (), ())),),
    )


def _run(model, tape, *, horizon=5.0, cutoff=10.0, delays=(0.0,) * 20,
         selector=select_canonical_count_space_bnb_frame):
    clock = SolveClock(*delays)
    result, selections = replay.simulate_delayed_trajectory(
        model, tape, horizon, cutoff, 36, selector, clock=clock,
    )
    return result, selections, clock


def test_arrivals_during_selection_wait_for_the_next_decision():
    result, _, clock = _run(
        _model(duration=1.0, offsets=(1.0,), lanes=2),
        ((0.0, 0, 0), (0.5, 0, 1)), horizon=1.0, cutoff=8.0,
        delays=(2.0, 2.0),
    )
    assert [frame["queue"] for frame in result["frames"]] == [(1, 0, 0), (1, 0, 0)]
    assert [frame["start"] for frame in result["frames"]] == [0.0, 3.0]
    assert [job["start"] for job in result["jobs"]] == [2.0, 5.0]
    assert [job["completion"] for job in result["jobs"]] == [3.0, 6.0]
    assert result["total_decision_delay_s"] == 4.0
    assert clock.calls == 4


def test_identical_released_histories_produce_identical_decision_prefixes():
    common = ((0.0, 0, 0), (0.5, 0, 1))
    first, _, _ = _run(_model(), common + ((3.1, 0, 2),), delays=(1.0,) * 10)
    second, _, _ = _run(_model(), common + ((3.2, 0, 2), (3.3, 0, 3)), delays=(1.0,) * 10)
    prefix = lambda result: [
        (frame["start"], frame["queue"], frame["option_indices"], frame["dispatch_time"])
        for frame in result["frames"] if frame["start"] < 3.1
    ]
    assert prefix(first) == prefix(second)


def test_zero_delay_reproduces_existing_trajectory_clock_and_job_outcomes():
    model = _model(duration=3.0, offsets=(1.0, 3.0), lanes=2)
    tape = ((0.25, 0, 0), (0.25, 0, 1), (1.5, 0, 2), (2.0, 0, 3))
    old = simulate_trajectory(model, tape, 3.0, 8.0, 36, select_canonical_count_space_bnb_frame)
    actual, selections, clock = _run(model, tape, horizon=3.0, cutoff=8.0)
    replay.audit_delayed_trajectory(model, actual, selections, ReferenceAudit(model))
    assert actual["jobs"] == old["jobs"]
    assert actual["snapshots"] == old["snapshots"]
    keys = ("start", "duration", "queue", "option_indices", "service")
    assert [[frame[key] for key in keys] for frame in actual["frames"]] == [
        [frame[key] for key in keys] for frame in old["frames"]]
    assert actual["total_decision_delay_s"] == 0.0
    assert actual["executed_frame_count"] == actual["decision_count"] == old["frame_count"]
    assert clock.calls == 2 * actual["decision_count"]


@pytest.mark.parametrize("delay", (1.0, 2.0))
def test_decision_finishing_at_or_after_cutoff_is_paid_but_never_dispatched(delay):
    model = _model()
    result, selections, clock = _run(
        model, ((0.0, 0, 0),), horizon=0.5, cutoff=1.0, delays=(delay,),
    )
    replay.audit_delayed_trajectory(model, result, selections, ReferenceAudit(model))
    assert result["decision_count"] == result["frame_count"] == 1
    assert result["executed_frame_count"] == 0
    assert result["selector_metrics"]["calls"] == 1
    assert result["frames"][0]["service"] == (1, 0, 0)
    assert result["frames"][0]["executed_service"] == (0, 0, 0)
    assert not result["frames"][0]["dispatched"]
    assert result["jobs"][0]["start"] is None
    assert result["snapshots"]["observation_end"]["queued_jobs"] == 1
    assert result["snapshots"]["observation_end"]["restricted_mean_flow_s"] == 1.0
    assert clock.calls == 2


def test_offsets_and_censoring_include_delay_once_and_preserve_inflight_jobs():
    result, _, _ = _run(
        _model(duration=10.0, offsets=(8.0, 2.0)),
        ((0.0, 0, 0), (0.0, 0, 1), (0.0, 0, 2), (1.0, 0, 3)),
        horizon=2.0, cutoff=5.0, delays=(1.0,),
    )
    snapshot = result["snapshots"]["observation_end"]
    assert [job["completion"] for job in result["jobs"]] == [3.0, 9.0, None, None]
    assert snapshot["completed_jobs"] == 1
    assert snapshot["inflight_jobs"] == 1
    assert snapshot["queued_jobs"] == 2
    assert snapshot["unfinished_jobs"] == 3
    assert snapshot["restricted_mean_flow_s"] == pytest.approx(17 / 4)
    assert result["last_frame_end"] == 11.0
    assert all(result["invariants"].values())


def test_idle_wait_is_one_second_after_solve_without_jumping_to_future_arrivals():
    result, _, _ = _run(
        _model(), ((0.25, 0, 0),), horizon=1.0, cutoff=4.0, delays=(0.5, 0.5),
    )
    assert result["frames"][0]["queue"] == (0, 0, 0)
    assert result["frames"][0]["duration"] == 1.0
    assert result["frames"][1]["start"] == 1.5
    assert result["jobs"][0]["start"] == 2.0
    assert result["jobs"][0]["completion"] == 4.0


def test_policy_is_called_again_at_repeated_queue_state_and_audit_is_deferred(monkeypatch):
    model, calls = _model(), []

    def selector(options, queue, lanes, budget):
        calls.append(tuple(queue))
        return select_canonical_count_space_bnb_frame(options, queue, lanes, budget)

    audit_calls = []
    original = ReferenceAudit.lookup

    def lookup(self, queue):
        audit_calls.append(tuple(queue))
        return original(self, queue)

    monkeypatch.setattr(ReferenceAudit, "lookup", lookup)
    result, selections, clock = _run(
        model, ((0.0, 0, 0), (0.5, 0, 1)), selector=selector, delays=(0.2, 0.2),
    )
    assert calls == [(1, 0, 0), (1, 0, 0)]
    assert audit_calls == []
    assert [job["completion"] for job in result["jobs"]] == pytest.approx([2.2, 4.4])
    clock_calls, simulated_end = clock.calls, result["last_frame_end"]
    audit = ReferenceAudit(model)
    replay.audit_delayed_trajectory(model, result, selections, audit)
    assert audit_calls == calls
    assert audit.summary()["actual_calls"] == 1
    assert result["selector_metrics"]["calls"] == 2
    assert result["total_decision_delay_s"] == pytest.approx(0.4)
    assert result["last_frame_end"] == simulated_end
    assert clock.calls == clock_calls


def test_false_certificate_is_retained_by_post_trajectory_audit():
    def invalid(options, queue, lanes, budget):
        selected = select_canonical_count_space_bnb_frame(options, queue, lanes, budget)
        return replace(selected, score=selected.score + 1.0)

    model = _model()
    result, selections, _ = _run(model, ((0.0, 0, 0),), selector=invalid)
    replay.audit_delayed_trajectory(model, result, selections, ReferenceAudit(model))
    assert result["selector_metrics"]["failures"] > 0
    assert result["snapshots"]["observation_end"]["completed_jobs"] == 1
