from copy import deepcopy
import json

import pytest

from algorithm.experiments.aligned_hardware_frame_comparison import (
    build_comparison, compare_reports,
)


def _reports():
    baseline = {
        "status": "PASS",
        "model": {"node": "registered", "lanes": 2},
        "protocol": {"loads": [0.8], "seeds": [41, 42], "budgets": [10, 120], "exact_reference_budget": 120},
        "certificates": [{"load_factor": 0.8, "effective_slack_eta": 0.1}],
        "runs": [],
    }
    for seed, factor in ((41, 1), (42, 2)):
        for budget in (10, 120):
            flow = (30 if budget == 10 else 10) * factor
            baseline["runs"].append({
                "load": 0.8, "seed": seed, "budget": budget,
                "job_count": 100, "completed_jobs": 100,
                "frame_count": 10, "exact_selection_frame_count": 10 if budget == 120 else 0,
                "mean_flow_s": flow, "p90_flow_s": flow * 2,
                "candidate_attempts": 100 if budget == 10 else 200,
                "bound_option_probes": 1000,
            })
    current = deepcopy(baseline)
    for row in current["runs"]:
        row["frame_count"] = 20
        row["exact_selection_frame_count"] = 20 if row["budget"] == 120 else 0
        row["mean_flow_s"] = (60 if row["seed"] == 41 else 90) if row["budget"] == 10 else 30
        row["p90_flow_s"] = 2 * row["mean_flow_s"]
        row["candidate_attempts"] = 80 if row["budget"] == 10 else 300
        row["bound_option_probes"] = 400
        row["bound_threshold_evaluations"] = 200
    return baseline, current


def test_ratios_use_paired_seeds_per_frame_costs_and_each_own_exact_reference():
    baseline, current = _reports()
    report = compare_reports(baseline, current)

    assert report["status"] == "PASS"  # Slower flow is a valid result.
    assert report["paired_run_count"] == 4
    first = report["paired_comparisons"][0]
    assert first["current_over_baseline_same_budget"]["mean_flow_s"] == 2
    assert first["baseline_over_own_exact"]["mean_flow_s"] == 3
    assert first["current_over_own_exact"]["mean_flow_s"] == 2
    assert first["current_over_baseline_same_budget"]["candidate_attempts_per_frame"] == pytest.approx(0.4)
    assert first["current_over_baseline_same_budget"]["bound_option_probes_per_frame"] == pytest.approx(0.2)
    assert first["current"]["bound_threshold_evaluations_per_frame"] == 10
    assert "bound_threshold_evaluations_per_frame" not in first["current_over_baseline_same_budget"]
    summary = report["summary_by_load_and_budget"][0]
    assert summary["current"]["mean_flow_s"] == {"mean": 75, "min": 60, "max": 90, "seed_count": 2}
    assert summary["current_over_baseline_same_budget"]["mean_flow_s"] == {
        "mean": 1.75, "min": 1.5, "max": 2, "seed_count": 2,
    }


@pytest.mark.parametrize("field", ("model", "protocol", "certificates"))
def test_changed_experiment_definition_is_refused(field):
    baseline, current = _reports()
    current[field] = {"changed": True}
    with pytest.raises(ValueError, match=f"{field} differs"):
        compare_reports(baseline, current)


def test_paired_job_counts_must_match_even_when_both_runs_completed():
    baseline, current = _reports()
    for row in current["runs"]:
        row["job_count"] = row["completed_jobs"] = 101
    with pytest.raises(ValueError, match="paired job counts differ"):
        compare_reports(baseline, current)


def test_missing_pair_is_refused():
    baseline, current = _reports()
    current["runs"].pop()
    with pytest.raises(ValueError, match="registered protocol matrix"):
        compare_reports(baseline, current)


def test_unfinished_population_and_inexact_reference_are_refused():
    baseline, current = _reports()
    current["runs"][0]["completed_jobs"] -= 1
    with pytest.raises(ValueError, match="incomplete job population"):
        compare_reports(baseline, current)
    current["runs"][0]["completed_jobs"] += 1
    current["runs"][1]["exact_selection_frame_count"] -= 1
    with pytest.raises(ValueError, match="reference search was not exact"):
        compare_reports(baseline, current)


def test_file_comparison_refusal_contains_no_partial_performance_results(tmp_path):
    baseline, current = _reports()
    current["model"]["lanes"] = 4
    before, after = tmp_path / "baseline.json", tmp_path / "current.json"
    before.write_text(json.dumps(baseline), encoding="utf-8")
    after.write_text(json.dumps(current), encoding="utf-8")

    report = build_comparison(baseline_path=before, current_path=after)

    assert report["status"] == "REFUSED"
    assert "model differs" in report["reason"]
    assert report["paired_comparisons"] == []
    assert report["summary_by_load_and_budget"] == []
