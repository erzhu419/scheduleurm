"""Prespecified seed-paired confirmation statistics for the aligned-frame study.

This module consumes the completed comparison; it neither runs experiments nor
selects budgets, seeds, margins, or endpoints from their observed performance.
"""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any

from scipy.stats import t as student_t


def summarize_confirmation(comparison: dict, registration: dict) -> dict:
    """Summarize four registered mean-flow claims with simultaneous t intervals.

    The estimand is the arithmetic mean of per-seed paired ratios. The primary
    family contains current/baseline improvement at the main load, followed by
    current/its-own-exact equivalence at each registered load. Comparison input
    validation and confirmation conclusions have separate statuses.
    """
    if comparison.get("status") != "PASS":
        raise ValueError("confirmation requires a completed PASS comparison")
    budget = registration["main_budget"]
    main_load = registration["main_load"]
    margin = float(registration["equivalence_margin"])
    family_alpha = float(registration["family_alpha"])
    family_size = registration["family_size"]
    loads = tuple(registration["protocol"]["loads"])
    seeds = tuple(registration["protocol"]["seeds"])
    if len(set(loads)) != len(loads) or main_load not in loads:
        raise ValueError("registered loads must be unique and include main_load")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("confirmation requires at least two distinct registered seeds")
    if family_size != 1 + len(loads):
        raise ValueError("family_size must equal one improvement claim plus all registered loads")
    if not math.isfinite(margin) or not 0.0 < margin < 1.0:
        raise ValueError("equivalence_margin must lie in (0, 1)")
    if not math.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")

    pairs: dict[tuple[Any, Any], dict] = {}
    for row in comparison["paired_comparisons"]:
        if row["budget"] != budget:
            continue
        key = (row["load"], row["seed"])
        if key in pairs:
            raise ValueError(f"duplicate main-budget paired result: {key}")
        pairs[key] = row
    expected = {(load, seed) for load in loads for seed in seeds}
    if set(pairs) != expected:
        raise ValueError("main-budget paired results do not match registered loads and seeds")

    n = len(seeds)
    interval_alpha = family_alpha / family_size
    confidence_level = 1.0 - interval_alpha
    critical = float(student_t.ppf(1.0 - interval_alpha / 2.0, df=n - 1))
    specifications = [
        ("improvement", main_load, "current_over_baseline_same_budget"),
        *(("equivalence", load, "current_over_own_exact") for load in loads),
    ]
    results = []
    for claim, load, ratio_source in specifications:
        ratios = []
        for seed in seeds:
            ratio = pairs[(load, seed)][ratio_source].get("mean_flow_s")
            if ratio is None or not math.isfinite(float(ratio)) or float(ratio) < 0.0:
                raise ValueError(f"undefined mean-flow paired ratio at {(load, seed, budget, ratio_source)}")
            ratios.append(float(ratio))
        estimate = mean(ratios)
        standard_error = stdev(ratios) / math.sqrt(n)
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
        if claim == "improvement":
            point_meets_claim = estimate < 1.0
            if upper < 1.0:
                conclusion, relation = "confirmed", "entirely_below_one"
            elif lower >= 1.0:
                conclusion, relation = "not_confirmed", "entirely_at_or_above_one"
            else:
                conclusion, relation = "inconclusive", "crosses_improvement_boundary"
            acceptance_interval = {"lower": None, "upper": 1.0, "upper_inclusive": False}
            claim_margin = 0.0
        else:
            band_lower, band_upper = 1.0 - margin, 1.0 + margin
            point_meets_claim = band_lower <= estimate <= band_upper
            if band_lower <= lower and upper <= band_upper:
                conclusion, relation = "confirmed", "entirely_within_equivalence_band"
            elif upper < band_lower:
                conclusion, relation = "not_confirmed", "entirely_below_equivalence_band"
            elif lower > band_upper:
                conclusion, relation = "not_confirmed", "entirely_above_equivalence_band"
            else:
                conclusion, relation = "inconclusive", "crosses_equivalence_boundary"
            acceptance_interval = {"lower": band_lower, "upper": band_upper, "upper_inclusive": True}
            claim_margin = margin
        results.append({
            "claim": claim,
            "load": load,
            "budget": budget,
            "endpoint": "mean_flow_s",
            "ratio_source": ratio_source,
            "n": n,
            "seeds": list(seeds),
            "ratios": ratios,
            "mean": estimate,
            "standard_error": standard_error,
            "confidence_interval": {"lower": lower, "upper": upper},
            "confidence_level": confidence_level,
            "degrees_of_freedom": n - 1,
            "critical_t": critical,
            "margin": claim_margin,
            "acceptance_interval": acceptance_interval,
            "point_estimate_meets_claim": point_meets_claim,
            "conclusion": conclusion,
            "interval_relation": relation,
        })
    return {
        "status": "PASS",
        "execution_status": comparison["status"],
        "statistical_analysis_status": "COMPLETE",
        "confirmation_required_for_execution_pass": False,
        "all_primary_claims_confirmed": all(row["conclusion"] == "confirmed" for row in results),
        "confirmed_claim_count": sum(row["conclusion"] == "confirmed" for row in results),
        "primary_comparisons": results,
        "family_alpha": family_alpha,
        "family_size": family_size,
        "per_comparison_two_sided_confidence_level": confidence_level,
        "estimator": "arithmetic mean of per-seed paired mean-flow ratios",
        "multiplicity": "Bonferroni simultaneous two-sided Student t intervals for the registered primary family",
        "statistical_limits": [
            "Seeds are the independent sampling units; jobs and frames are not treated as independent replicates.",
            "Small-sample Student t intervals for seed-level ratios rely on the usual approximately normal sampling assumption; they are not distribution-free guarantees.",
            "Failure of an interval to lie inside its acceptance region does not by itself establish performance degradation. A ratio below the equivalence band means faster mean flow, while still failing the registered closeness claim.",
            "The equivalence estimand is abs(mean(seed_ratio) - 1), not mean(abs(seed_ratio - 1)), and not a ratio of pooled mean-flow values.",
        ],
    }
