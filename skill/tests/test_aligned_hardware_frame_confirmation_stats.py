from __future__ import annotations

from copy import deepcopy
import math

import pytest

from algorithm.experiments.aligned_hardware_frame_confirmation_stats import summarize_confirmation


def _inputs(improvement=None, exact=None):
    seeds = list(range(1041, 1051))
    loads = [0.5, 0.8, 0.95]
    registration = {
        "main_budget": 26,
        "main_load": 0.5,
        "equivalence_margin": 0.002,
        "family_alpha": 0.05,
        "family_size": 4,
        "protocol": {"loads": loads, "seeds": seeds},
    }
    improvement = improvement or [0.9] * len(seeds)
    exact = exact or {load: [1.0] * len(seeds) for load in loads}
    pairs = []
    for load in loads:
        for index, seed in enumerate(seeds):
            pairs.append({
                "load": load,
                "seed": seed,
                "budget": 26,
                "current_over_baseline_same_budget": {"mean_flow_s": improvement[index]},
                "current_over_own_exact": {"mean_flow_s": exact[load][index]},
                # These intentionally differ to detect ratio-of-means substitution.
                "current": {"mean_flow_s": 100.0 * (index + 1)},
                "baseline": {"mean_flow_s": 100.0},
            })
    return {"status": "PASS", "paired_comparisons": pairs}, registration


def test_arithmetic_paired_ratios_and_adjusted_interval_use_registered_seed_units():
    ratios = [0.90 + index * 0.01 for index in range(10)]
    comparison, registration = _inputs(improvement=ratios)
    # Input order must not redefine registered seed order or sample membership.
    comparison["paired_comparisons"].reverse()
    result = summarize_confirmation(comparison, registration)
    effect = result["primary_comparisons"][0]
    expected_mean = sum(ratios) / 10
    expected_se = math.sqrt(sum((value - expected_mean) ** 2 for value in ratios) / 9 / 10)
    assert effect["ratios"] == ratios
    assert effect["seeds"] == registration["protocol"]["seeds"]
    assert effect["n"] == 10
    assert effect["mean"] == pytest.approx(expected_mean)
    assert effect["standard_error"] == pytest.approx(expected_se)
    assert effect["confidence_level"] == 0.9875
    assert effect["degrees_of_freedom"] == 9
    # Independent distribution inversion: P(t_9 > critical) = .00625.
    # This checks the two-sided Bonferroni tail, rather than a one-sided .0125 tail.
    from scipy.special import betainc
    critical = effect["critical_t"]
    assert 0.5 * betainc(4.5, 0.5, 9 / (9 + critical * critical)) == pytest.approx(0.00625)
    assert effect["confidence_interval"] == pytest.approx({
        "lower": expected_mean - critical * expected_se,
        "upper": expected_mean + critical * expected_se,
    })


def test_zero_variance_claims_confirm_without_changing_execution_status():
    comparison, registration = _inputs()
    result = summarize_confirmation(comparison, registration)
    assert result["status"] == result["execution_status"] == "PASS"
    assert result["confirmed_claim_count"] == 4
    assert result["all_primary_claims_confirmed"]
    assert all(row["standard_error"] == 0 for row in result["primary_comparisons"])
    assert result["confirmation_required_for_execution_pass"] is False


def test_point_estimate_in_band_with_wide_interval_is_inconclusive_not_degradation():
    exact = {load: [0.99, 1.01] * 5 for load in (0.5, 0.8, 0.95)}
    comparison, registration = _inputs(improvement=[0.8, 1.1] * 5, exact=exact)
    result = summarize_confirmation(comparison, registration)
    assert result["status"] == "PASS"
    assert not result["all_primary_claims_confirmed"]
    assert all(row["point_estimate_meets_claim"] for row in result["primary_comparisons"])
    assert all(row["conclusion"] == "inconclusive" for row in result["primary_comparisons"])


def test_interval_below_equivalence_band_is_not_mislabeled_as_degradation():
    exact = {0.5: [0.99] * 10, 0.8: [1.01] * 10, 0.95: [1.0] * 10}
    comparison, registration = _inputs(improvement=[1.1] * 10, exact=exact)
    result = summarize_confirmation(comparison, registration)
    improvement, faster, slower, equal = result["primary_comparisons"]
    assert improvement["conclusion"] == "not_confirmed"
    assert faster["conclusion"] == slower["conclusion"] == "not_confirmed"
    assert faster["interval_relation"] == "entirely_below_equivalence_band"
    assert slower["interval_relation"] == "entirely_above_equivalence_band"
    assert equal["conclusion"] == "confirmed"
    assert result["status"] == "PASS"


def test_equivalence_is_absolute_mean_ratio_difference_not_mean_absolute_difference():
    exact = {load: [0.999, 1.001] * 5 for load in (0.5, 0.8, 0.95)}
    comparison, registration = _inputs(exact=exact)
    result = summarize_confirmation(comparison, registration)
    for row in result["primary_comparisons"][1:]:
        assert row["mean"] == 1.0
        assert row["margin"] == 0.002
        assert row["acceptance_interval"]["lower"] == 0.998
        assert row["acceptance_interval"]["upper"] == 1.002
        assert row["conclusion"] == "confirmed"


@pytest.mark.parametrize("mutation,match", [
    ("missing", "registered loads and seeds"),
    ("extra", "registered loads and seeds"),
    ("duplicate", "duplicate main-budget"),
    ("null_ratio", "undefined mean-flow"),
    ("nan_ratio", "undefined mean-flow"),
    ("failed_execution", "PASS comparison"),
    ("changed_family", "family_size"),
])
def test_missing_invalid_or_changed_primary_population_is_not_silently_dropped(mutation, match):
    comparison, registration = _inputs()
    if mutation == "missing":
        comparison["paired_comparisons"].pop()
    elif mutation == "extra":
        extra = deepcopy(comparison["paired_comparisons"][0])
        extra["seed"] = 41
        comparison["paired_comparisons"].append(extra)
    elif mutation == "duplicate":
        comparison["paired_comparisons"].append(comparison["paired_comparisons"][0])
    elif mutation in ("null_ratio", "nan_ratio"):
        comparison["paired_comparisons"][0]["current_over_own_exact"]["mean_flow_s"] = (
            None if mutation == "null_ratio" else float("nan")
        )
    elif mutation == "failed_execution":
        comparison["status"] = "REFUSED"
    elif mutation == "changed_family":
        registration["family_size"] = 7
    with pytest.raises(ValueError, match=match):
        summarize_confirmation(comparison, registration)
