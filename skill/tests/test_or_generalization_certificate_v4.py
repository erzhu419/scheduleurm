from __future__ import annotations

from algorithm.experiments.or_generalization_certificate_v4 import (
    build_certificate,
    markdown_report,
)


def test_v4_retains_new_protocols_and_negative_results():
    report = build_certificate()

    assert report["gate"]["pass"] is True
    assert report["gate"]["all_negative_results_retained"] is True
    assert report["gate"]["performance_superiority_claim_ready"] is False
    assert report["fjsp_exact_ledger_confirmation_v2"][
        "fixed_budget_cp_sat_dominates_count"
    ] == 14
    assert report["fjsp_replication_audit"][
        "strict_full_artifact_reproducibility_ready"
    ] is False
    assert report["mmrcpsp_renewal_stream"]["run_count"] == 105
    assert report["mmrcpsp_renewal_stream"][
        "stochastic_stability_claim_ready"
    ] is False
    assert report["port_statewise_trajectory_v3"][
        "industrial_deployment_claim_ready"
    ] is False
    rendered = markdown_report(report)
    assert "CP-SAT dominates 14/20" in rendered
    assert "finite drift only" in rendered
    assert "no industrial deployment" in rendered
