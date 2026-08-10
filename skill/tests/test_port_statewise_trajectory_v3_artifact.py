from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.port_statewise_trajectory_v3_artifact import (
    PortStatewiseArtifactError,
    markdown_report,
    write_outputs,
)


def _artifact() -> dict:
    return {
        "artifact_sha256_excluding_self": "a" * 64,
        "report": {
            "status": "PORT_STATEWISE_TRAJECTORY_V3_PASS",
            "pass": True,
            "certificates": {
                "bacasp_source_core": {"ready": True},
                "synthetic_migration": {
                    "ready": True,
                    "selection": {
                        "oracle_gap_alpha0": 0.0,
                        "finite_family_penalty_bound_P0": 2.0,
                        "selected_candidate": {"candidate_id": "robust"},
                    },
                },
            },
            "claim_boundary": {
                "supports": ["finite registered actions"],
                "does_not_support": ["industrial deployment"],
            },
        },
    }


def test_markdown_keeps_source_and_synthetic_evidence_distinct():
    rendered = markdown_report(_artifact())

    assert "BACASP source-core ready: `true`" in rendered
    assert "Synthetic statewise migration ready: `true`" in rendered
    assert "non-substitutable" in rendered
    assert "industrial deployment" in rendered


def test_writer_refuses_to_overwrite_results(tmp_path: Path):
    output = tmp_path / "artifact.json"
    markdown = tmp_path / "artifact.md"
    outputs = write_outputs(_artifact(), json_path=output, markdown_path=markdown)

    assert outputs["json_sha256"]
    assert outputs["markdown_sha256"]
    with pytest.raises(PortStatewiseArtifactError, match="refusing to overwrite"):
        write_outputs(_artifact(), json_path=output, markdown_path=markdown)
