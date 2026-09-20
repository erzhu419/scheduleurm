"""Synthetic seed-level ratios for the registered support-cap confirmation API."""
from copy import deepcopy
import math

import pytest
from scipy.special import betainc

from algorithm.experiments.aligned_hardware_support_cap_confirmation_stats import summarize_confirmation


def _inputs(ratios_by_load=None):
    # These values are synthetic test inputs, not generated experiment outcomes.
    seeds = list(range(2041, 2061))
    loads = [0.5, 0.8, 0.95]
    registration = {
        "protocol": {"seeds": seeds, "loads": loads, "budgets": [10, 14, 26, 120]},
        "main_budget": 26,
        "main_policy": "holding_cap_1pct",
        "primary_loads": [0.8, 0.95],
        "family_alpha": 0.05,
        "family_size": 2,
    }
    ratios_by_load = ratios_by_load or {load: [0.9] * 20 for load in loads}
    rows = []
    for load in loads:
        for index, seed in enumerate(seeds):
            ratio = ratios_by_load[load][index]
            baseline = 10.0 * (index + 1)
            rows.append({
                "load": load, "seed": seed, "budget": 26, "policy": "holding_cap_1pct",
                "waiting_over_v4_same_budget": {"mean_flow_s": ratio},
                "baseline": {"mean_flow_s": baseline},
                "waiting_rule": {"mean_flow_s": ratio * baseline},
            })
    return {"status": "PASS", "paired_comparisons": rows}, registration


def test_paired_arithmetic_ratios_and_independent_bonferroni_tail():
    ratios = [0.8 + 0.01 * index for index in range(20)]
    comparison, registration = _inputs({load: ratios for load in (0.5, 0.8, 0.95)})
    comparison["paired_comparisons"].reverse()
    summary = summarize_confirmation(comparison, registration)
    expected_mean = sum(ratios) / 20
    expected_se = math.sqrt(sum((value - expected_mean) ** 2 for value in ratios) / 19 / 20)
    pooled_ratio = sum(ratio * (index + 1) for index, ratio in enumerate(ratios)) / sum(range(1, 21))
    assert pooled_ratio != pytest.approx(expected_mean)
    assert summary["per_comparison_two_sided_confidence_level"] == 0.975
    for claim in summary["primary_comparisons"]:
        assert claim["ratios"] == ratios
        assert claim["seeds"] == registration["protocol"]["seeds"]
        assert claim["n"] == 20
        assert claim["degrees_of_freedom"] == 19
        assert claim["mean"] == pytest.approx(expected_mean)
        assert claim["standard_error"] == pytest.approx(expected_se)
        assert claim["confidence_level"] == 0.975
        # Independent t-tail evaluation using the incomplete beta function:
        # family alpha .05 / 2 claims / 2 tails = .0125, not .025.
        critical = claim["critical_t"]
        assert critical > 0
        assert 0.5 * betainc(9.5, 0.5, 19 / (19 + critical * critical)) == pytest.approx(0.0125)
        assert claim["confidence_interval"] == pytest.approx({
            "lower": expected_mean - critical * expected_se,
            "upper": expected_mean + critical * expected_se,
        })


def test_only_main_policy_budget_and_two_primary_loads_enter_the_family():
    comparison, registration = _inputs({
        0.5: [2.0] * 20, 0.8: [0.9] * 20, 0.95: [0.95] * 20,
    })
    distractors = []
    for row in comparison["paired_comparisons"]:
        control = deepcopy(row)
        control["policy"] = "holding_cap_0pct"
        control["waiting_over_v4_same_budget"]["mean_flow_s"] = 20.0
        distractors.append(control)
        exact_budget = deepcopy(row)
        exact_budget["budget"] = 120
        exact_budget["waiting_over_v4_same_budget"]["mean_flow_s"] = 30.0
        distractors.append(exact_budget)
    comparison["paired_comparisons"].extend(distractors)
    summary = summarize_confirmation(comparison, registration)
    claims = summary["primary_comparisons"]
    assert len(claims) == summary["family_size"] == 2
    assert [claim["load"] for claim in claims] == [0.8, 0.95]
    assert [claim["mean"] for claim in claims] == pytest.approx([0.9, 0.95])
    assert all(claim["policy"] == "holding_cap_1pct" and claim["budget"] == 26 for claim in claims)
    assert all(claim["claim"] == "improvement" for claim in claims)
    assert all(claim["ratio_source"] == "waiting_over_v4_same_budget" for claim in claims)


def test_zero_variance_below_one_confirms_but_equality_does_not():
    comparison, registration = _inputs({
        0.5: [0.9] * 20, 0.8: [0.9] * 20, 0.95: [1.0] * 20,
    })
    summary = summarize_confirmation(comparison, registration)
    improved, equal = summary["primary_comparisons"]
    assert improved["standard_error"] == equal["standard_error"] == 0.0
    assert improved["conclusion"] == "confirmed"
    assert equal["conclusion"] == "not_confirmed"
    assert equal["confidence_interval"] == {"lower": 1.0, "upper": 1.0}
    assert not equal["point_estimate_meets_claim"]
    assert summary["confirmed_claim_count"] == 1
    assert not summary["all_primary_claims_confirmed"]
    assert summary["status"] == summary["execution_status"] == comparison["status"] == "PASS"
    assert summary["statistical_analysis_status"] == "COMPLETE"
    assert summary["confirmation_required_for_execution_pass"] is False


def test_interval_crossing_one_is_inconclusive_even_with_a_favorable_mean():
    comparison, registration = _inputs({load: [0.8, 1.18] * 10 for load in (0.5, 0.8, 0.95)})
    summary = summarize_confirmation(comparison, registration)
    for claim in summary["primary_comparisons"]:
        assert claim["mean"] == pytest.approx(0.99)
        assert claim["point_estimate_meets_claim"]
        assert claim["confidence_interval"]["lower"] < 1 < claim["confidence_interval"]["upper"]
        assert claim["conclusion"] == "inconclusive"
    assert summary["status"] == summary["execution_status"] == "PASS"
    assert summary["confirmed_claim_count"] == 0
    assert not summary["all_primary_claims_confirmed"]


@pytest.mark.parametrize("mutation", (
    "missing_primary", "missing_secondary", "duplicate_primary",
    "duplicate_secondary", "unregistered_seed", "failed_execution",
))
def test_missing_or_duplicate_members_are_not_silently_dropped(mutation):
    comparison, registration = _inputs()
    rows = comparison["paired_comparisons"]
    primary = next(row for row in rows if row["load"] == 0.8)
    secondary = next(row for row in rows if row["load"] == 0.5)
    if mutation == "missing_primary":
        rows.remove(primary)
    elif mutation == "missing_secondary":
        rows.remove(secondary)
    elif mutation == "duplicate_primary":
        rows.append(deepcopy(primary))
    elif mutation == "duplicate_secondary":
        rows.append(deepcopy(secondary))
    elif mutation == "unregistered_seed":
        extra = deepcopy(primary)
        extra["seed"] = 41
        rows.append(extra)
    else:
        comparison["status"] = "FAIL"
    with pytest.raises(ValueError):
        summarize_confirmation(comparison, registration)
