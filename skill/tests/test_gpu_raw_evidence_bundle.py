from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from algorithm.experiments.gpu_raw_evidence_bundle import (
    EvidenceBundleError,
    build_gpu_raw_evidence_bundle,
)


def test_bundle_is_deterministic_and_contains_reachable_raw_files(tmp_path: Path) -> None:
    repo, run_root, root = _fixture(tmp_path)
    first = _build(tmp_path / "first", root=root, repo=repo, run_root=run_root)
    second = _build(tmp_path / "second", root=root, repo=repo, run_root=run_root)

    assert first["status"] == "PASS"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["entry_count"] == 5
    bundle = Path(first["bundle_path"])
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == first["bundle_sha256"]
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    assert names == {
        "manifest.json",
        "repository/artifacts/child.json",
        "repository/artifacts/root.json",
        "scheduler_runs/run-1/reports/summary.json",
        "scheduler_runs/run-1/raw/progress.log",
        "scheduler_runs/run-1/raw/sidecar.log",
    }


def test_missing_referenced_scheduler_file_fails_closed(tmp_path: Path) -> None:
    repo, run_root, root = _fixture(tmp_path)
    (run_root / "run-1" / "raw" / "progress.log").unlink()

    with pytest.raises(EvidenceBundleError, match="scheduler evidence is missing"):
        _build(tmp_path / "failed", root=root, repo=repo, run_root=run_root)


def test_nonpassing_root_is_rejected(tmp_path: Path) -> None:
    repo, run_root, root = _fixture(tmp_path)
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["status"] = "WAIT_INPUTS"
    payload["pass"] = False
    root.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceBundleError, match="root artifact is not PASS"):
        _build(tmp_path / "failed", root=root, repo=repo, run_root=run_root)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    artifacts = repo / "artifacts"
    run_root = tmp_path / "scheduler_runs"
    raw = run_root / "run-1" / "raw"
    artifacts.mkdir(parents=True)
    raw.mkdir(parents=True)
    progress = raw / "progress.log"
    sidecar = raw / "sidecar.log"
    progress.write_text("ScheduleurmProgress current=1 total=2\n", encoding="utf-8")
    sidecar.write_text("gpu=0 util=80\n", encoding="utf-8")
    summary = run_root / "run-1" / "reports" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"log_path": str(progress), "gpu_monitor_log_path": str(sidecar)}),
        encoding="utf-8",
    )
    child = artifacts / "child.json"
    child.write_text(
        json.dumps({"summary_path": str(summary)}),
        encoding="utf-8",
    )
    root = artifacts / "root.json"
    root.write_text(
        json.dumps(
            {
                "gate": "fixture_gate",
                "status": "PASS",
                "pass": True,
                "ordinary_running_tasks_touched": False,
                "source_artifacts": [{"path": str(child)}],
            }
        ),
        encoding="utf-8",
    )
    return repo, run_root, root


def _build(output_root: Path, *, root: Path, repo: Path, run_root: Path):
    output_root.mkdir(parents=True)
    return build_gpu_raw_evidence_bundle(
        root_artifacts=[root],
        output_path=output_root / "bundle.tar.gz",
        manifest_path=output_root / "bundle.json",
        markdown_path=output_root / "bundle.md",
        repo_root=repo,
        scheduler_run_root=run_root,
    )
