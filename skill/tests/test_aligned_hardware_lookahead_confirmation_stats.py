"""Synthetic paired ratios for improvement and zero-margin no-worsening."""
from copy import deepcopy
import math

import pytest
from scipy.special import betainc

from algorithm.experiments.aligned_hardware_lookahead_confirmation_stats import summarize_confirmation


def _inputs(ratios_by_load=None):
    # Registered seed labels identify synthetic test rows only. No arrival tape
    # or measured outcome is generated or read by these tests.
    seeds = list(range(3041, 3061))
    loads = [0.5, 0.8, 0.95]
    registration = {
        "protocol": {"seeds": seeds, "loads": loads, "budgets": [10, 14, 26, 120]},
        "main_budget": 26,
        "main_policy": "lookahead_cap_1pct",
        "primary_claims": [
            {"load": 0.8, "claim": "improvement"},
            {"load": 0.95, "claim": "improvement"},
            {"load": 0.5, "claim": "no_worsening"},
        ],
        "family_alpha": 0.05,
        "family_size": 3,
    }
    ratios_by_load = ratios_by_load or {load: [0.9] * 20 for load in loads}
    rows = []
    for load in loads:
        for index, seed in enumerate(seeds):
            ratio = ratios_by_load[load][index]
            baseline = 10.0 * (index + 1)
            rows.append({
                "load": load, "seed": seed, "budget": 26, "policy": "lookahead_cap_1pct",
                "waiting_over_v4_same_budget": {"mean_flow_s": ratio},
                "baseline": {"mean_flow_s": baseline},
                "waiting_rule": {"mean_flow_s": ratio * baseline},
            })
    return {"status": "PASS", "paired_comparisons": rows}, registration


def test_paired_ratio_estimator_and_independent_three_claim_bonferroni_tail():
    ratios = [0.8 + 0.01 * index for index in range(20)]
    comparison, registration = _inputs({load: ratios for load in (0.5, 0.8, 0.95)})
    comparison["paired_comparisons"].reverse()
    result = summarize_confirmation(comparison, registration)
    expected_mean = sum(ratios) / 20
    expected_se = math.sqrt(sum((ratio - expected_mean) ** 2 for ratio in ratios) / 19 / 20)
    pooled_ratio = sum(ratio * (index + 1) for index, ratio in enumerate(ratios)) / sum(range(1, 21))
    assert pooled_ratio != pytest.approx(expected_mean)
    confidence = 1.0 - 0.05 / 3.0
    assert result["per_comparison_two_sided_confidence_level"] == pytest.approx(confidence)
    for claim in result["primary_comparisons"]:
        assert claim["ratios"] == ratios
        assert claim["seeds"] == registration["protocol"]["seeds"]
        assert claim["n"] == 20
        assert claim["degrees_of_freedom"] == 19
        assert claim["mean"] == pytest.approx(expected_mean)
        assert claim["standard_error"] == pytest.approx(expected_se)
        assert claim["confidence_level"] == pytest.approx(confidence)
        critical = claim["critical_t"]
        assert critical > 0
        # Independently evaluate the t19 upper tail via incomplete beta.
        assert 0.5 * betainc(9.5, 0.5, 19 / (19 + critical ** 2)) == pytest.approx(0.05 / 3 / 2)
        assert claim["confidence_interval"] == pytest.approx({
            "lower": expected_mean - critical * expected_se,
            "upper": expected_mean + critical * expected_se,
        })


def test_claim_order_main_policy_and_main_budget_are_preserved():
    comparison, registration = _inputs({
        0.5: [0.98] * 20, 0.8: [0.9] * 20, 0.95: [0.95] * 20,
    })
    extras = []
    for row in comparison["paired_comparisons"]:
        control = deepcopy(row)
        control["policy"] = "lookahead_area_only"
        control["waiting_over_v4_same_budget"]["mean_flow_s"] = 20.0
        extras.append(control)
        exact_budget = deepcopy(row)
        exact_budget["budget"] = 120
        exact_budget["waiting_over_v4_same_budget"]["mean_flow_s"] = 30.0
        extras.append(exact_budget)
    comparison["paired_comparisons"].extend(extras)
    result = summarize_confirmation(comparison, registration)
    claims = result["primary_comparisons"]
    assert [(claim["load"], claim["claim"]) for claim in claims] == [
        (0.8, "improvement"), (0.95, "improvement"), (0.5, "no_worsening"),
    ]
    assert len(claims) == result["family_size"] == 3
    assert [claim["mean"] for claim in claims] == pytest.approx([0.9, 0.95, 0.98])
    assert all(claim["policy"] == "lookahead_cap_1pct" and claim["budget"] == 26 for claim in claims)
    assert all(claim["ratio_source"] == "waiting_over_v4_same_budget" for claim in claims)


def test_all_one_ratios_confirm_no_worsening_but_not_strict_improvement():
    comparison, registration = _inputs({load: [1.0] * 20 for load in (0.5, 0.8, 0.95)})
    result = summarize_confirmation(comparison, registration)
    first, second, low = result["primary_comparisons"]
    for claim in (first, second, low):
        assert claim["standard_error"] == 0.0
        assert claim["confidence_interval"] == {"lower": 1.0, "upper": 1.0}
    assert first["conclusion"] == second["conclusion"] == "not_confirmed"
    assert not first["point_estimate_meets_claim"]
    assert low["conclusion"] == "confirmed"
    assert low["point_estimate_meets_claim"]
    assert not result["all_improvement_claims_confirmed"]
    assert result["low_load_no_worsening_confirmed"]
    assert not result["all_primary_claims_confirmed"]
    assert result["confirmed_claim_count"] == 1
    assert result["status"] == result["execution_status"] == comparison["status"] == "PASS"
    assert result["confirmation_required_for_execution_pass"] is False


def test_high_load_improvement_cannot_override_low_load_regression():
    comparison, registration = _inputs({
        0.5: [1.0001] * 20, 0.8: [0.9] * 20, 0.95: [0.95] * 20,
    })
    result = summarize_confirmation(comparison, registration)
    low = result["primary_comparisons"][2]
    assert result["all_improvement_claims_confirmed"]
    assert low["conclusion"] == "not_confirmed"
    assert not low["point_estimate_meets_claim"]
    assert not result["low_load_no_worsening_confirmed"]
    assert not result["all_primary_claims_confirmed"]
    assert result["confirmed_claim_count"] == 2
    # No unregistered tolerance margin may accept even this small increase.
    assert low["confidence_interval"]["lower"] > 1.0
    assert result["status"] == result["execution_status"] == "PASS"


def test_intervals_crossing_one_are_inconclusive_not_degradation():
    comparison, registration = _inputs({load: [0.8, 1.18] * 10 for load in (0.5, 0.8, 0.95)})
    result = summarize_confirmation(comparison, registration)
    for claim in result["primary_comparisons"]:
        assert claim["mean"] == pytest.approx(0.99)
        assert claim["point_estimate_meets_claim"]
        assert claim["confidence_interval"]["lower"] < 1 < claim["confidence_interval"]["upper"]
        assert claim["conclusion"] == "inconclusive"
    assert result["confirmed_claim_count"] == 0
    assert not result["all_primary_claims_confirmed"]
    assert result["status"] == result["execution_status"] == "PASS"
    assert result["statistical_analysis_status"] == "COMPLETE"


@pytest.mark.parametrize("load", (0.5, 0.8, 0.95))
@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_every_registered_load_rejects_missing_and_duplicate_members(load, mutation):
    comparison, registration = _inputs()
    rows = comparison["paired_comparisons"]
    member = next(row for row in rows if row["load"] == load)
    if mutation == "missing":
        rows.remove(member)
    else:
        rows.append(deepcopy(member))
    with pytest.raises(ValueError):
        summarize_confirmation(comparison, registration)


def test_unregistered_seed_is_not_silently_added_to_the_sample():
    comparison, registration = _inputs()
    extra = deepcopy(comparison["paired_comparisons"][0])
    extra["seed"] = 41
    comparison["paired_comparisons"].append(extra)
    with pytest.raises(ValueError):
        summarize_confirmation(comparison, registration)
