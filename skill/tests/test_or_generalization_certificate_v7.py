from __future__ import annotations

from algorithm.experiments.or_generalization_certificate_v7 import (
    build_certificate,
    latex_macros,
    markdown_report,
)


def test_v7_admits_solver_actions_and_retains_cross_domain_failures():
    report = build_certificate()

    assert report["gate"]["pass"] is True
    assert report["gate"]["all_negative_results_retained"] is True
    assert report["gate"]["performance_superiority_claim_ready"] is False

    fjsp = report["fjsp"]["prospective_solver_action_union_holdout"]
    assert fjsp["instance_count"] == 20
    assert fjsp["exact_union_argmax_count"] == 20
    assert fjsp["solver_relation_counts"] == {"same_action": 20}
    assert fjsp["global_fjsp_optimality_claim_ready"] is False

    mm = report["mmrcpsp"]["renewal_stream_total_queue"]
    assert mm["baseline_dominates_count"] == 21
    assert mm["retained_negative_result"] is True

    port_negative = report["port"]["prospective_statewise_external_holdout"]
    assert port_negative["instance_count"] == 54
    assert port_negative["selected_pareto_nondominated_count"] == 27
    assert port_negative["retained_negative_result"] is True

    port_union = report["port"]["prospective_finite_action_union_holdout"]
    assert port_union["instance_count"] == 54
    assert port_union["performance_superiority_claim_ready"] is False

    text = markdown_report(report)
    assert "negative result retained" in text
    assert "not CP-SAT superiority" in text

    tex = latex_macros(report)
    assert r"\newcommand{\FjspUnionSameSolverCount}{20}" in tex
    assert r"\newcommand{\PortStatewiseNondominatedCount}{27}" in tex
    assert r"\newcommand{\PortUnionNondominatedCount}" in tex
