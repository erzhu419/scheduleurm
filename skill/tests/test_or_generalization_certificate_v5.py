from __future__ import annotations

from algorithm.experiments.or_generalization_certificate_v5 import (
    build_certificate,
    markdown_report,
)


def test_v5_retains_counterexamples_tradeoffs_and_reproducibility():
    report = build_certificate()

    assert report["gate"]["pass"] is True
    assert report["gate"]["all_negative_results_retained"] is True
    assert report["gate"]["performance_superiority_claim_ready"] is False
    assert report["fjsp"]["fixed_budget_cp_sat_dominates_count"] == 14
    mm = report["mmrcpsp"]
    assert mm["v1_total_queue_result"]["baseline_dominates_count"] == 21
    assert mm["v1_total_queue_result"]["ours_nondominated_count"] == 0
    holdout = mm["prospective_class_balance_arrival_tape_holdout"]
    assert holdout["ours_nondominated_count"] == 35
    assert holdout["ours_strict_every_baseline_count"] == 0
    assert holdout["new_structural_instance_holdout"] is False
    assert mm["reproducibility_audit"][
        "semantic_reproducibility_ready_count"
    ] == 2
    assert mm["reproducibility_audit"][
        "path_independent_gzip_claim_ready"
    ] is False
    assert report["port"]["source_core"]["robust_maxweight_dominated_by"] == [
        "reconfiguration_greedy",
        "spt_static",
    ]
    assert report["port"]["synthetic_migration"][
        "duration_normalized_oracle_gap_alpha0"
    ] == 0.0
    assert report["port"]["industrial_deployment_claim_ready"] is False
    rendered = markdown_report(report)
    assert "baseline dominates 21/21" in rendered
    assert "ours nondominated 35/35; strict-all 0/35" in rendered
    assert "source robust MaxWeight dominated" in rendered
    assert "no industrial deployment" in rendered
