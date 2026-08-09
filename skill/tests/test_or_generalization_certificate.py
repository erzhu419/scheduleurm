from __future__ import annotations

import json

from algorithm.experiments.or_generalization_certificate import (
    build_certificate,
    markdown_report,
    write_artifacts,
)


def test_generalization_certificate_preserves_component_boundaries():
    report = build_certificate()
    assert report["gate"] == {
        "pass": True,
        "status": "OR_GENERALIZATION_CERTIFICATE_PASS",
        "component_count": 3,
        "all_component_protocol_gates_auditable": True,
        "all_successful_selected_policies_pareto_nondominated": True,
        "arbitrary_domain_optimality_claim_ready": False,
        "stochastic_stability_transfers_without_domain_model": False,
    }
    assert report["fjsp"]["ours_pareto_nondominated_count"] == 4
    assert report["mmrcpsp"]["v1_prospective_gate_pass"] is False
    assert report["mmrcpsp"]["repair_regression"][
        "prospective_external_holdout_claim_ready"
    ] is False
    assert report["mmrcpsp"]["v2_ours_pareto_nondominated_count"] == 56
    assert report["port"][
        "selected_pareto_nondominated_on_every_registered_instance"
    ] is True
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
    assert json.loads(first_json)["gate"]["pass"] is True
    assert "MMRCPSP prospective repair protocol" in markdown_report(report)
