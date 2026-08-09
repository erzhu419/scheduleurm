from __future__ import annotations

import json

from algorithm.experiments.or_generalization_certificate import (
    build_certificate,
    markdown_report,
    write_artifacts,
)


def test_generalization_certificate_retains_positive_and_negative_evidence():
    report = build_certificate()
    gate = report["gate"]
    assert gate["pass"] is True
    assert gate["all_registered_outcomes_retained"] is True
    assert gate["all_component_protocols_auditable"] is True
    assert gate["all_selected_policies_pareto_nondominated"] is False
    assert gate["performance_superiority_claim_ready"] is False
    assert gate["arbitrary_domain_optimality_claim_ready"] is False

    assert report["fjsp"]["first_family_holdout"]["protocol_pass"] is False
    assert report["fjsp"]["first_family_holdout"]["pareto_nondominated_count"] == 19
    assert report["fjsp"]["hurink_confirmation"]["protocol_pass"] is True
    assert report["fjsp"]["hurink_confirmation"]["pareto_nondominated_count"] == 19

    assert report["mmrcpsp"]["v1_failure_retained"] is True
    assert report["mmrcpsp"]["opened_row_repair"]["prospective_claim_ready"] is False
    assert report["mmrcpsp"]["multi_family_confirmation"]["instance_count"] == 25
    assert report["mmrcpsp"]["multi_family_confirmation"]["pareto_nondominated_count"] == 25

    factor = report["port"]["r85_r86_factor_policy_confirmation"]
    assert factor["protocol_pass"] is True
    assert factor["candidate_generation_on_holdout"] is False
    assert factor["holdout_feedback_used_for_selection"] is False
    assert report["port"]["performance_superiority_claim_ready"] is False
    assert report["artifact_sha256_excluding_self"]


def test_generalization_artifacts_are_byte_deterministic(tmp_path):
    report = build_certificate()
    json_path = tmp_path / "gate.json"
    markdown_path = tmp_path / "gate.md"
    write_artifacts(report, json_path, markdown_path)
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    write_artifacts(report, json_path, markdown_path)
    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    payload = json.loads(first_json)
    assert payload["gate"]["pass"] is True
    assert payload["gate"]["performance_superiority_claim_ready"] is False
    markdown = markdown_report(report)
    assert "Retained negative evidence" in markdown
    assert "19/20" in markdown
