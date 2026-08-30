from __future__ import annotations

import json
from pathlib import Path

from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_solver_action_union import FROZEN_CONFIG
from algorithm.experiments.fjsp_solver_action_union_holdout import (
    PREREGISTRATION_SCHEMA,
    _digest,
    verify_preregistration,
)
from algorithm.experiments.fjsp_benchmark import POLICY_ORDER


ROOT = Path(__file__).resolve().parents[2]


def test_preregistration_binds_source_config_and_implementation(tmp_path: Path):
    source = ROOT / "tests/data/fjsp_brandimarte_tiny.fjs"
    local = tmp_path / "tiny.fjs"
    local.write_bytes(source.read_bytes())
    instance = load_fjsp_instance(local)
    implementation = ROOT / "algorithm/experiments/fjsp_solver_action_union.py"
    import hashlib

    prereg = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "holdout_outcomes_observed_before_freeze": False,
        "solver_action_config": FROZEN_CONFIG.snapshot(),
        "baseline_policies": list(POLICY_ORDER),
        "implementation_file_sha256": {
            "algorithm/experiments/fjsp_solver_action_union.py": hashlib.sha256(
                implementation.read_bytes()
            ).hexdigest()
        },
        "selected_instances": [
            {
                "local_path": str(local),
                "file_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                "canonical_source_sha256": instance.source_sha256,
            }
        ],
    }
    prereg["preregistration_sha256_excluding_self"] = _digest(prereg)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg), encoding="utf-8")

    report = verify_preregistration(path)
    assert report["ready"] is True
    assert report["instance_count"] == 1
