from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm.experiments.fjsp_external_holdout import (
    EXPECTED_INSTANCE_IDS,
    ExternalHoldoutError,
    build_external_holdout_report,
    validate_external_source,
    verify_pre_analysis_freeze,
)


def test_pre_analysis_freeze_is_repository_auditable() -> None:
    audit = verify_pre_analysis_freeze()
    assert audit["pass"] is True
    assert audit["external_suite_absent_from_freeze_commit"] is True
    assert len(audit["implementation_files"]) == 3


def test_external_source_is_hash_locked_and_complete() -> None:
    gate = validate_external_source()
    assert gate["pass"] is True
    assert tuple(row["id"] for row in gate["verified_instances"]) == EXPECTED_INSTANCE_IDS


def test_external_source_rejects_tampered_bytes(tmp_path: Path) -> None:
    source_gate = validate_external_source()
    original_dir = Path("tests/data/fjsp_kacem_external_holdout")
    for row in source_gate["verified_instances"]:
        (tmp_path / row["file"]).write_bytes((original_dir / row["file"]).read_bytes())
    manifest = json.loads((original_dir / "source_manifest.json").read_text(encoding="utf-8"))
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "kacem1.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n1\n")
    with pytest.raises(ExternalHoldoutError, match="source hash mismatch"):
        validate_external_source(tmp_path, manifest_path)


def test_frozen_policy_runs_complete_external_holdout() -> None:
    report = build_external_holdout_report()
    assert report["gate"]["pass"] is True
    assert report["gate"]["prospective_external_holdout_ready"] is True
    assert report["gate"]["instance_count"] == 4
    assert report["gate"]["ours_instance_pareto_nondominated_count"] == 4
    assert all(
        row["trajectory_search"]["exact_family_argmax"]
        and row["trajectory_search"]["oracle_gap_within_generated_family"] == 0
        for row in report["instances"]
    )
