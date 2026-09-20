"""Frozen per-load confirmation statistics for the mixed-successor candidate.

Arrival seeds are the independent units. Planning repeats are averaged inside
each arrival seed, and Bonferroni Student-t intervals cover the three load
claims. Execution integrity and the candidate's adoption decision are separate.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, stdev

from scipy.stats import t


V4_POLICY = "v4_support"
ANALYTIC_POLICY = "lookahead_cap_1pct"
RAW_POLICY = "poisson_rollout_cap_1pct"
MIXED_POLICY = "mixed_successor_rollout_cap_1pct"
CLAIMS = ((.5, "no_worsening"), (.8, "improvement"), (.95, "improvement"))
ADOPTION_CRITERIA = {
    "version": "mixed_successor_per_load_confirmation_v1",
    "independent_unit": "real arrival seed; 20 units per load",
    "planning_aggregation": "arithmetic mean of three matched mixed/raw ratios inside each load and real arrival seed",
    "primary_metric": "mean flow time through completion of the full arrival population",
    "primary_claims": [{"load": load, "claim": claim} for load, claim in CLAIMS],
    "family_alpha": .05, "family_size": 3,
    "interval": "Bonferroni simultaneous two-sided Student-t confidence intervals, df=19",
    "critical_t_quantile": 1 - .05 / (2 * 3),
    "improvement_rule": "upper confidence limit < 1",
    "no_worsening_rule": "upper confidence limit <= 1; zero noninferiority margin",
    "primary_confirmation": "all three per-load claims must pass",
    "adoption_guardrails": {
        "each_load_mean_mixed_over_v4_flow_ratio_max": 1.0,
        "each_load_mean_mixed_over_raw_p90_ratio_max": 1.0,
        "interpretation": "predeclared point-estimate adoption thresholds, not additional statistical noninferiority claims",
    },
    "analytic_comparison": "descriptive",
    "planning_seed_gate": None,
    "adoption": "execution integrity, all three primary claims, and every point-estimate guardrail must pass",
    "inference_scope": "three frozen loads under the model and the three frozen planning streams; Student-t coverage uses the usual arrival-seed sampling assumptions",
}


def _primary_passes(load, upper):
    return upper <= 1.0 if load == .5 else upper < 1.0


def summarize_confirmation(runs, loads, seeds, planning_seeds):
    """Match the full roster before computing 20-unit intervals at each load."""
    loads, seeds, planning_seeds = tuple(loads), tuple(seeds), tuple(planning_seeds)
    if (loads != tuple(load for load, _ in CLAIMS) or len(seeds) != 20
            or len(set(seeds)) != 20 or len(planning_seeds) != 3 or len(set(planning_seeds)) != 3):
        raise ValueError("confirmation requires the registered three loads, 20 real seeds and three planning seeds")
    roster = ((V4_POLICY, None), (ANALYTIC_POLICY, None)) + tuple(
        (policy, planning_seed) for policy in (RAW_POLICY, MIXED_POLICY) for planning_seed in planning_seeds)
    expected = {(load, seed, policy, planning_seed)
                for load in loads for seed in seeds for policy, planning_seed in roster}
    indexed = {}
    for row in runs:
        key = (row["load"], row["seed"], row["policy"], row["planning_seed"])
        if key in indexed:
            raise ValueError("duplicate confirmation run prevents unique planning-seed matching")
        if row["budget"] != 26:
            raise ValueError("confirmation budget differs from registered K26")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("confirmation roster must contain every matched planning repetition and both deterministic controls")

    paired, by_load = [], []
    critical = float(t.ppf(ADOPTION_CRITERIA["critical_t_quantile"], 19))
    for load, claim in CLAIMS:
        by_seed = []
        for seed in seeds:
            v4 = indexed[(load, seed, V4_POLICY, None)]["metrics"]
            analytic = indexed[(load, seed, ANALYTIC_POLICY, None)]["metrics"]
            seed_pairs = []
            for planning_seed in planning_seeds:
                raw = indexed[(load, seed, RAW_POLICY, planning_seed)]["metrics"]
                mixed = indexed[(load, seed, MIXED_POLICY, planning_seed)]["metrics"]
                pair = {
                    "load": load, "seed": seed, "planning_seed": planning_seed,
                    "flow_ratio_mixed_over_raw": mixed["mean_flow_s"] / raw["mean_flow_s"],
                    "flow_ratio_mixed_over_v4": mixed["mean_flow_s"] / v4["mean_flow_s"],
                    "flow_ratio_mixed_over_analytic": mixed["mean_flow_s"] / analytic["mean_flow_s"],
                    "p90_ratio_mixed_over_raw": mixed["p90_flow_s"] / raw["p90_flow_s"],
                }
                seed_pairs.append(pair)
                paired.append(pair)
            by_seed.append({
                "seed": seed,
                **{f"mean_{key}": mean(pair[key] for pair in seed_pairs) for key in (
                    "flow_ratio_mixed_over_raw", "flow_ratio_mixed_over_v4",
                    "flow_ratio_mixed_over_analytic", "p90_ratio_mixed_over_raw",
                )},
            })
        values = [row["mean_flow_ratio_mixed_over_raw"] for row in by_seed]
        average, deviation = mean(values), stdev(values)
        se = deviation / sqrt(len(values))
        lower, upper = average - critical * se, average + critical * se
        by_load.append({
            "load": load, "arrival_seed_count": len(values), "planning_seeds": list(planning_seeds),
            "by_arrival_seed": by_seed,
            "primary": {
                "claim": claim, "mean_ratio": average, "sample_standard_deviation": deviation,
                "standard_error": se, "df": 19, "critical_t": critical,
                "individual_confidence_level": 1 - .05 / 3,
                "ci_lower": lower, "ci_upper": upper, "passed": _primary_passes(load, upper),
            },
            **{key: mean(row[key] for row in by_seed) for key in (
                "mean_flow_ratio_mixed_over_v4", "mean_flow_ratio_mixed_over_analytic",
                "mean_p90_ratio_mixed_over_raw",
            )},
        })
    return {
        "run_count": len(indexed), "paired_run_count": len(paired),
        "arrival_seed_count": len(seeds), "seeds": list(seeds), "planning_seeds": list(planning_seeds),
        "family_alpha": .05, "family_size": 3,
        "primary_confirmed": all(row["primary"]["passed"] for row in by_load),
        "paired_runs": paired, "by_load": by_load,
    }


def adoption_decision(summary, integrity_checks):
    """Apply only the frozen gates; a failed performance gate is not execution failure."""
    if [row["load"] for row in summary["by_load"]] != [load for load, _ in CLAIMS]:
        raise ValueError("adoption decision requires all three registered load summaries")
    by_load = [{
        "load": row["load"],
        "primary_passed": _primary_passes(row["load"], row["primary"]["ci_upper"]),
        "v4_flow_guardrail_passed": row["mean_flow_ratio_mixed_over_v4"] <= 1.0,
        "p90_guardrail_passed": row["mean_p90_ratio_mixed_over_raw"] <= 1.0,
    } for row in summary["by_load"]]
    integrity_passed = bool(integrity_checks) and all(integrity_checks.values())
    primary_confirmed = all(row["primary_passed"] for row in by_load)
    guardrails_passed = all(row["v4_flow_guardrail_passed"] and row["p90_guardrail_passed"] for row in by_load)
    adopted = integrity_passed and primary_confirmed and guardrails_passed
    return {
        "status": "ADOPT" if adopted else "NOT_ADOPT", "adopted": adopted,
        "integrity_passed": integrity_passed, "primary_confirmed": primary_confirmed,
        "guardrails_passed": guardrails_passed, "by_load": by_load,
    }
