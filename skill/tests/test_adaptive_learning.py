from algorithm.adaptive_learning import (
    BucketCandidate,
    RoundRobinActiveBucketSampler,
    TwoWindowMeanShiftDetector,
    detector_false_alarm_bound,
    detector_miss_bound,
    sampler_hoeffding_radius,
)
from algorithm.placement import available_policies, load_placement_policy


def test_round_robin_sampler_covers_active_buckets_before_exploitation():
    sampler = RoundRobinActiveBucketSampler(min_samples_per_bucket=3)
    candidates = [
        BucketCandidate(bucket="b", score=10.0, payload="b"),
        BucketCandidate(bucket="a", score=1.0, payload="a"),
    ]

    decisions = [sampler.choose(candidates) for _ in range(6)]

    assert all(decision.forced_exploration for decision in decisions)
    assert sampler.counts() == {"a": 3, "b": 3}
    cert = sampler.coverage_certificate(["a", "b"])
    assert cert["coverage_certified"] is True
    assert cert["coverage_probability_lower_bound"] == 1.0


def test_round_robin_sampler_uses_score_after_coverage():
    sampler = RoundRobinActiveBucketSampler(min_samples_per_bucket=1)
    candidates = [
        {"candidate_bucket": "a", "score": 1.0},
        {"candidate_bucket": "b", "score": 5.0},
    ]

    sampler.choose(candidates)
    sampler.choose(candidates)
    decision = sampler.choose(candidates)

    assert decision.forced_exploration is False
    assert decision.selected.bucket == "b"


def test_two_window_detector_reports_bounded_mean_shift():
    detector = TwoWindowMeanShiftDetector(window_size=4, threshold=0.4, min_shift=0.8)
    events = []
    for value in [0.0] * 4 + [1.0] * 4:
        event = detector.update("r0", value)
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert events[0].shift == 1.0
    cert = detector.tail_certificate(switching_count=1, failure_budget=1.0)
    assert cert["detection_delay_bound_decisions"] == 8


def test_probability_bound_helpers_are_monotone_enough_for_certificate_defaults():
    radius = sampler_hoeffding_radius(
        sample_count=1561,
        active_bucket_count=64,
        failure_budget=0.025,
    )
    miss = detector_miss_bound(window_size=512, threshold=0.2, min_shift=0.4)
    false_alarm = detector_false_alarm_bound(window_size=512, threshold=0.2)

    assert radius < 0.06
    assert miss < 0.0001
    assert false_alarm < 0.0001


def test_adaptive_theorem_policy_is_opt_in_and_legacy_remains_default(monkeypatch):
    monkeypatch.delenv("SCHEDULEURM_ALGORITHM", raising=False)

    assert "adaptive_theorem_maxweight_v1" in set(available_policies())
    assert load_placement_policy().name == "legacy"
    assert load_placement_policy("adaptive_theorem_maxweight_v1").name == "adaptive_theorem_maxweight_v1"
