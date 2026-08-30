from __future__ import annotations

import json
from pathlib import Path

from algorithm.experiments.port_public_action_union import (
    OUTER_SCORE_CONFIG,
    frozen_action_family,
)
from algorithm.experiments.port_public_action_union_holdout import (
    verify_preregistration,
)
from algorithm.experiments.port_public_external_holdout import _digest
from algorithm.experiments.port_scheduling_benchmark import BASELINE_POLICIES
from algorithm.experiments.port_trajectory_upgrade import SOURCE_COST_METRICS


def test_action_union_preregistration_verifier_binds_family_and_implementation(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    source = Path(__file__).resolve().parents[2] / "algorithm/experiments/port_public_action_union.py"
    import hashlib

    prereg = {
        "schema_version": "scheduleurm.port_action_union_preregistration.v1",
        "holdout_outcomes_observed_before_freeze": False,
        "suite_manifest_file_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_rule": {"external_holdout_replications": [81, 82]},
        "frozen_action_family": [plan.snapshot() for plan in frozen_action_family()],
        "outer_score_config": OUTER_SCORE_CONFIG.snapshot(),
        "implementation_file_sha256": {
            "algorithm/experiments/port_public_action_union.py": hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        },
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
    }
    prereg["preregistration_sha256_excluding_self"] = _digest(prereg)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg), encoding="utf-8")

    report = verify_preregistration(path, manifest)
    assert report["ready"] is True
    assert report["action_family_size"] == len(frozen_action_family())
