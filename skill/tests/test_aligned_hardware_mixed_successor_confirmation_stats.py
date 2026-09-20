"""Independent arithmetic-ratio inference for the registered 20-seed units."""
from copy import deepcopy
import math

import pytest
from scipy.special import betainc

from algorithm.experiments.aligned_hardware_mixed_successor_confirmation_stats import (
    adoption_decision, summarize_confirmation,
)


LOADS = (.5, .8, .95)
SEEDS = tuple(range(4041, 4061))
PLANNING_SEEDS = (101, 102, 103)
RAW = "poisson_rollout_cap_1pct"
MIXED = "mixed_successor_rollout_cap_1pct"


def _runs(varying=False):
    """Synthetic metric values; no arrival tapes or confirmation RNGs are drawn."""
    rows = []

    def append(load, seed, policy, planning_seed, flow, p90):
        rows.append({"load": load, "seed": seed, "budget": 26, "policy": policy,
                     "planning_seed": planning_seed,
                     "metrics": {"mean_flow_s": flow, "p90_flow_s": p90,
                                 "job_count": 10, "completed_jobs": 10}})

    for load_index, load in enumerate(LOADS):
        for seed_index, seed in enumerate(SEEDS):
            for policy in ("v4_support", "lookahead_cap_1pct"):
                append(load, seed, policy, None, 1e6, 2e6)
            for planning_index, planning_seed in enumerate(PLANNING_SEEDS):
                baseline = 10. * (seed_index + 1) * (load_index + 1) * 2 ** planning_index
                ratio = (.8 + .005 * seed_index + .02 * load_index + (-.2, 0., .2)[planning_index]
                         if varying else .9)
                append(load, seed, RAW, planning_seed, baseline, 2 * baseline)
                append(load, seed, MIXED, planning_seed, baseline * ratio, 2 * baseline)
    return rows


def _summary(varying=False):
    rows = _runs(varying)
    rows.reverse()  # Matching must use identity, never position in the input.
    return summarize_confirmation(rows, LOADS, SEEDS, PLANNING_SEEDS)


def test_planning_ratios_average_before_seed_inference_with_three_bonferroni_tails():
    summary = _summary(varying=True)
    assert summary["arrival_seed_count"] == 20 and len(summary["paired_runs"]) == 180
    assert summary["primary_confirmed"]
    for load_index, row in enumerate(summary["by_load"]):
        assert row["load"] == LOADS[load_index]
        expected = [.8 + .005 * index + .02 * load_index for index in range(20)]
        by_seed, primary = row["by_arrival_seed"], row["primary"]
        assert [member["seed"] for member in by_seed] == list(SEEDS)
        assert [member["mean_flow_ratio_mixed_over_raw"] for member in by_seed] == pytest.approx(expected)
        estimate = sum(expected) / 20
        deviation = math.sqrt(sum((value - estimate) ** 2 for value in expected) / 19)
        error = deviation / math.sqrt(20)
        assert primary["df"] == 19
        assert primary["mean_ratio"] == pytest.approx(estimate)
        assert primary["sample_standard_deviation"] == pytest.approx(deviation)
        assert primary["standard_error"] == pytest.approx(error)
        critical = primary["critical_t"]
        # An independent Student tail identity checks alpha .05 / 3 / 2,
        # ruling out a single interval, one-sided tail, or 60-row df.
        assert .5 * betainc(9.5, .5, 19 / (19 + critical ** 2)) == pytest.approx(.05 / 6)
        assert primary["ci_lower"] == pytest.approx(estimate - critical * error)
        assert primary["ci_upper"] == pytest.approx(estimate + critical * error)
        assert primary["passed"]
        naive = [ratio + delta for ratio in expected for delta in (-.2, 0., .2)]
        naive_se = math.sqrt(sum((value - estimate) ** 2 for value in naive) / 59 / 60)
        assert primary["standard_error"] != pytest.approx(naive_se)
        # A ratio of pooled planning flows weights planning seeds by1,2,4.
        pooled_planning_ratio = sum((expected[0] + delta) * weight
                                    for delta, weight in zip((-.2, 0., .2), (1, 2, 4))) / 7
        assert by_seed[0]["mean_flow_ratio_mixed_over_raw"] != pytest.approx(pooled_planning_ratio)
        assert row["mean_p90_ratio_mixed_over_raw"] == 1.


@pytest.mark.parametrize("load,field,value,adopted", (
    (.5, "ci_upper", 1., True),
    (.5, "ci_upper", 1.000000001, False),
    (.8, "ci_upper", 1., False),
    (.95, "ci_upper", 1., False),
    (.8, "mean_flow_ratio_mixed_over_v4", 1., True),
    (.8, "mean_flow_ratio_mixed_over_v4", 1.000000001, False),
    (.95, "mean_p90_ratio_mixed_over_raw", 1., True),
    (.95, "mean_p90_ratio_mixed_over_raw", 1.000000001, False),
))
def test_adoption_recomputes_strict_and_inclusive_boundaries(load, field, value, adopted):
    summary = _summary()
    target = next(row for row in summary["by_load"] if row["load"] == load)
    if field == "ci_upper":
        target["primary"][field] = value
    else:
        target[field] = value
    # Analytic performance is descriptive. Cached conclusions must not
    # override the registered numeric boundaries after a recomputation.
    for row in summary["by_load"]:
        row["mean_flow_ratio_mixed_over_analytic"] = 100.
        row["primary"]["passed"] = True
    summary["primary_confirmed"] = True
    result = adoption_decision(summary, {"complete_roster": True, "all_jobs_complete": True})
    assert result["adopted"] is adopted
    assert result["status"] == ("ADOPT" if adopted else "NOT_ADOPT")
    assert result["integrity_passed"]


def test_failed_execution_blocks_adoption_and_cached_failure_does_not_override_numeric_pass():
    summary = _summary()
    for row in summary["by_load"]:
        row["primary"]["passed"] = False
    summary["primary_confirmed"] = False
    assert adoption_decision(summary, {"complete_roster": True})["status"] == "ADOPT"
    result = adoption_decision(summary, {"complete_roster": False})
    assert result["status"] == "NOT_ADOPT" and not result["adopted"]
    assert not result["integrity_passed"]


@pytest.mark.parametrize("mutation", ("missing_planning", "duplicate_planning", "wrong_planning", "missing_control"))
def test_incomplete_or_duplicate_registered_roster_is_not_silently_reweighted(mutation):
    rows = _runs()
    member = next(row for row in rows if row["policy"] == MIXED)
    if mutation == "missing_planning":
        rows.remove(member)
    elif mutation == "duplicate_planning":
        rows.append(deepcopy(member))
    elif mutation == "wrong_planning":
        member["planning_seed"] = 0
    else:
        rows.remove(next(row for row in rows if row["policy"] == "v4_support"))
    with pytest.raises(ValueError):
        summarize_confirmation(rows, LOADS, SEEDS, PLANNING_SEEDS)
