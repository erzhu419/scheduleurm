from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from algorithm.experiments.mmrcpsp_external_holdout import (
    ExternalHoldoutError,
    build_external_holdout_report,
    frozen_protocol_sha256,
    validate_external_source,
    verify_pre_analysis_freeze,
)


SUITE = Path("tests/data/mmrcpsp_psplib_external_holdout")


def test_policy_freeze_and_preregistration_are_repository_auditable():
    audit = verify_pre_analysis_freeze()
    assert audit["pass"] is True
    assert audit["selection_registered_after_policy_freeze"] is True
    assert audit["instance_bytes_absent_at_preregistration"] is True
    assert len(audit["implementation_files"]) == 4
    assert frozen_protocol_sha256() == audit["protocol_config_sha256"]


def test_external_source_is_hash_locked_and_matches_preregistered_selection():
    gate = validate_external_source()
    assert gate["pass"] is True
    assert len(gate["verified_instances"]) == 56
    preregistration = json.loads((SUITE / "preregistration.json").read_text())
    assert tuple(row["file"] for row in gate["verified_instances"]) == tuple(
        preregistration["selection"]["selected_members"]
    )


def test_external_source_rejects_tampered_instance(tmp_path: Path):
    root = tmp_path / "suite"
    shutil.copytree(SUITE, root)
    target = next(root.glob("*.mm"))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ExternalHoldoutError, match="source hash mismatch"):
        validate_external_source(
            root, root / "preregistration.json", root / "source_manifest.json"
        )


def test_external_source_rejects_selection_drift(tmp_path: Path):
    root = tmp_path / "suite"
    shutil.copytree(SUITE, root)
    manifest = json.loads((root / "source_manifest.json").read_text())
    manifest["instances"] = manifest["instances"][:-1]
    (root / "source_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExternalHoldoutError, match="preregistered selection"):
        validate_external_source(
            root, root / "preregistration.json", root / "source_manifest.json"
        )


def test_frozen_policy_fails_closed_on_first_preregistered_counterexample():
    report = build_external_holdout_report()
    gate = report["gate"]
    assert gate["pass"] is False
    assert gate["prospective_external_holdout_ready"] is False
    assert gate["instance_count"] == 0
    assert gate["registered_instance_count"] == 56
    assert gate["completed_all_registered_instances"] is False
    assert gate["all_schedules_feasible"] is False
    assert report["first_failure"] == {
        "instance": "j102_10",
        "exception_type": "TrajectoryUpgradeError",
        "reason": (
            "j102_10: rollout seed scheduleurm_robust_maxweight is infeasible: "
            "resource_deadlock:eligible=[6, 9]:nonrenewable_consumed=[27, 20]"
        ),
        "fail_closed": True,
    }
    assert gate["global_mmrcpsp_optimality_claim"] is False
    assert gate["stochastic_throughput_optimality_claim"] is False
