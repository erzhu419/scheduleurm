from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from algorithm.experiments.mmrcpsp_external_holdout import ExternalHoldoutError
from algorithm.experiments.mmrcpsp_external_holdout_v2 import (
    build_external_holdout_report,
    validate_external_source,
    verify_pre_analysis_freeze,
)


SUITE = Path("tests/data/mmrcpsp_psplib_external_holdout_v2")


def test_repair_freeze_and_v2_preregistration_are_auditable():
    audit = verify_pre_analysis_freeze()
    assert audit["pass"] is True
    assert audit["selection_registered_after_repair_freeze"] is True
    assert audit["instance_bytes_absent_at_preregistration"] is True
    assert len(audit["implementation_files"]) == 4


def test_v2_source_is_hash_locked_and_disjoint():
    gate = validate_external_source()
    assert gate["pass"] is True
    assert gate["disjoint_from_v1_and_development"] is True
    assert len(gate["verified_instances"]) == 56


def test_v2_source_rejects_tampering(tmp_path: Path):
    root = tmp_path / "suite"
    shutil.copytree(SUITE, root)
    target = next(root.glob("*.mm"))
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ExternalHoldoutError, match="source hash mismatch"):
        validate_external_source(
            root, root / "preregistration.json", root / "source_manifest.json"
        )


def test_repaired_policy_runs_full_disjoint_holdout():
    report = build_external_holdout_report(workers=8)
    gate = report["gate"]
    assert gate["pass"] is True
    assert gate["prospective_external_holdout_ready"] is True
    assert gate["registered_instance_count"] == 56
    assert gate["completed_instance_count"] == 56
    assert gate["completed_all_registered_instances"] is True
    assert gate["all_schedules_feasible"] is True
    assert gate["zero_resource_violation"] is True
    assert gate["zero_precedence_violation"] is True
    assert gate["exact_generated_family_argmax_on_every_instance"] is True
    assert gate["global_mmrcpsp_optimality_claim"] is False
    assert gate["stochastic_throughput_optimality_claim"] is False
    assert report["failures"] == []
