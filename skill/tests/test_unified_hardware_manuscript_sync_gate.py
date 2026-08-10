from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algorithm.experiments.unified_hardware_manuscript_sync_gate import (
    REQUIRED_MANUSCRIPT_TOKENS,
    build_manuscript_sync_report,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    replay_path = tmp_path / "replay.json"
    results_path = tmp_path / "results.json"
    figure_path = tmp_path / "figure.json"
    tex_path = tmp_path / "results.tex"
    manuscript_path = tmp_path / "main.tex"
    replay = {
        "gate": "unified_hardware_or_replay",
        "schema_version": 2,
        "status": "PASS",
        "pass": True,
        "comparison_kind": "same-cache_policy-semantics",
        "full_stack_external_binary_comparison": False,
        "run_count": 28,
        "policy_count": 7,
        "scenarios": [
            {"scenario_id": "q00:a", "scenario_kind": "hardware_local"},
            {"scenario_id": "q01:b", "scenario_kind": "hardware_local"},
            {
                "scenario_id": "migration:q11:p50",
                "scenario_kind": "controlled_migration_counterfactual",
            },
        ],
    }
    _write(replay_path, replay)
    replay_sha256 = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    _write(
        results_path,
        {
            "gate": "unified_hardware_paper_results",
            "status": "PASS",
            "pass": True,
            "replay_sha256": replay_sha256,
        },
    )
    _write(
        figure_path,
        {
            "source_artifact_sha256": replay_sha256,
            "source_gate": "unified_hardware_or_replay",
            "source_schema_version": 2,
            "source_run_count": 28,
            "source_policy_count": 7,
            "included_hardware_scenario_ids": ["q00:a", "q01:b"],
        },
    )
    _write(tex_path, f"% Source replay SHA256: {replay_sha256}\n")
    _write(manuscript_path, "\n".join(REQUIRED_MANUSCRIPT_TOKENS))
    return {
        "replay_path": replay_path,
        "results_path": results_path,
        "figure_data_path": figure_path,
        "tex_path": tex_path,
        "manuscript_path": manuscript_path,
    }


def test_all_paper_surfaces_must_share_one_replay_hash(tmp_path: Path) -> None:
    report = build_manuscript_sync_report(**_inputs(tmp_path))

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert all(report["checks"].values())


def test_stale_figure_or_missing_manuscript_macro_fails_closed(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    figure = json.loads(paths["figure_data_path"].read_text(encoding="utf-8"))
    figure["source_artifact_sha256"] = "0" * 64
    _write(paths["figure_data_path"], figure)
    manuscript = paths["manuscript_path"].read_text(encoding="utf-8")
    paths["manuscript_path"].write_text(
        manuscript.replace(r"\UnifiedSotaPolicyRows", ""), encoding="utf-8"
    )

    report = build_manuscript_sync_report(**paths)

    assert report["status"] == "FAIL_SYNC"
    assert set(report["failed_checks"]) == {
        "figure_replay_hash_matches",
        "manuscript_consumes_generated_results",
    }
