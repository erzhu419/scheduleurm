from __future__ import annotations

from algorithm.experiments.or_generalization_certificate_v6 import (
    build_certificate,
    markdown_report,
)


def test_v6_restores_structural_holdout_and_keeps_all_boundaries():
    report = build_certificate()

    assert report["gate"]["pass"] is True
    assert report["gate"]["all_negative_results_retained"] is True
    assert report["gate"]["deterministic_virtual_frame_bridge_ready"] is True
    assert report["gate"]["performance_superiority_claim_ready"] is False
    assert report["gate"]["physical_port_stability_claim_ready"] is False

    structural = report["mmrcpsp"]["prospective_structural_family_holdout"]
    assert structural["instance_count"] == 25
    assert structural["ours_pareto_nondominated_count"] == 25
    assert structural["ours_strict_every_baseline_count"] == 11
    assert structural["ours_attains_public_optimum_count"] == 7

    total = report["mmrcpsp"]["renewal_stream_total_queue"]
    assert total["baseline_dominates_count"] == 21
    port = report["port"]["registered_trajectory_refresh"]
    assert port["trajectory_context_pair_count"] == 124
    assert port["prospective_holdout_claim_ready"] is False
    assert "negative result retained" in markdown_report(report)
