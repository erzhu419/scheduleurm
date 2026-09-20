"""Registered seed-paired improvement tests for the fixed support-cap policy.

The module consumes completed comparisons only. Registered policy, budget,
loads and seeds determine membership before any confidence interval is formed.
"""
from __future__ import annotations

import math
from statistics import mean, stdev

from scipy.stats import t as student_t


def summarize_confirmation(comparison: dict, registration: dict) -> dict:
    """Separate successful execution from the registered improvement claims."""
    if comparison.get("status") != "PASS":
        raise ValueError("confirmation requires a completed PASS comparison")
    policy = registration["main_policy"]
    budget = registration["main_budget"]
    protocol = registration["protocol"]
    loads = tuple(protocol["loads"])
    seeds = tuple(protocol["seeds"])
    primary_loads = tuple(registration["primary_loads"])
    family_alpha = float(registration["family_alpha"])
    family_size = registration["family_size"]
    if not isinstance(policy, str) or not policy:
        raise ValueError("main_policy must name the registered policy")
    if budget not in protocol["budgets"]:
        raise ValueError("main_budget must belong to the registered budgets")
    if not loads or len(set(loads)) != len(loads):
        raise ValueError("registered loads must be nonempty and unique")
    if (not primary_loads or len(set(primary_loads)) != len(primary_loads)
            or not set(primary_loads).issubset(loads)):
        raise ValueError("primary_loads must be unique members of the registered loads")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("confirmation requires at least two distinct registered seeds")
    if family_size != len(primary_loads):
        raise ValueError("family_size must equal the number of registered primary loads")
    if not math.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")

    ratio_source = "waiting_over_v4_same_budget"
    pairs = {}
    for row in comparison["paired_comparisons"]:
        if row["policy"] != policy or row["budget"] != budget:
            continue
        key = (row["load"], row["seed"])
        if key in pairs:
            raise ValueError(f"duplicate main-policy main-budget paired result: {key}")
        ratio = row.get(ratio_source, {}).get("mean_flow_s")
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) < 0.0:
            raise ValueError(f"undefined mean-flow paired ratio at {(policy, budget, *key)}")
        pairs[key] = float(ratio)
    expected = {(load, seed) for load in loads for seed in seeds}
    if set(pairs) != expected:
        raise ValueError("main-policy main-budget paired results do not match all registered loads and seeds")

    n = len(seeds)
    interval_alpha = family_alpha / family_size
    confidence_level = 1.0 - interval_alpha
    critical = float(student_t.ppf(1.0 - interval_alpha / 2.0, df=n - 1))
    results = []
    for load in primary_loads:
        ratios = [pairs[(load, seed)] for seed in seeds]
        estimate = mean(ratios)
        standard_error = stdev(ratios) / math.sqrt(n)
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
        if upper < 1.0:
            conclusion, relation = "confirmed", "entirely_below_one"
        elif lower >= 1.0:
            conclusion, relation = "not_confirmed", "entirely_at_or_above_one"
        else:
            conclusion, relation = "inconclusive", "crosses_improvement_boundary"
        results.append({
            "claim": "improvement", "policy": policy, "load": load, "budget": budget,
            "endpoint": "mean_flow_s", "ratio_source": ratio_source,
            "n": n, "seeds": list(seeds), "ratios": ratios,
            "mean": estimate, "standard_error": standard_error,
            "confidence_interval": {"lower": lower, "upper": upper},
            "confidence_level": confidence_level,
            "degrees_of_freedom": n - 1, "critical_t": critical,
            "margin": 0.0,
            "acceptance_interval": {"lower": None, "upper": 1.0, "upper_inclusive": False},
            "point_estimate_meets_claim": estimate < 1.0,
            "conclusion": conclusion, "interval_relation": relation,
        })
    return {
        "status": "PASS",
        "execution_status": comparison["status"],
        "statistical_analysis_status": "COMPLETE",
        "confirmation_required_for_execution_pass": False,
        "all_primary_claims_confirmed": all(row["conclusion"] == "confirmed" for row in results),
        "confirmed_claim_count": sum(row["conclusion"] == "confirmed" for row in results),
        "primary_comparisons": results,
        "family_alpha": family_alpha, "family_size": family_size,
        "per_comparison_two_sided_confidence_level": confidence_level,
        "estimator": "arithmetic mean of per-seed paired mean-flow ratios",
        "multiplicity": "Bonferroni simultaneous two-sided Student t intervals for the registered primary loads",
        "statistical_limits": [
            "Arrival seeds are the independent sampling units; jobs and frames are not independent replicates.",
            "Small-sample Student t intervals rely on approximately normal sampling of seed-level paired ratios and are not distribution-free guarantees.",
            "The endpoint is the arithmetic mean of paired ratios against v4 at the same budget, not a ratio of pooled mean flows.",
        ],
    }
