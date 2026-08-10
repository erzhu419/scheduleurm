from __future__ import annotations

import json
from pathlib import Path
import subprocess

from algorithm.experiments.final_eta_or_closure_pipeline import (
    Stage,
    build_stages,
    preflight_sources,
    run_pipeline,
)


def _gate(path: Path, *, passing: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "gate": path.stem,
                "status": "PASS" if passing else "WAIT_MEASUREMENTS",
                "pass": passing,
            }
        ),
        encoding="utf-8",
    )


def test_preflight_reports_every_missing_or_nonpassing_source(tmp_path: Path) -> None:
    passing = tmp_path / "passing.json"
    waiting = tmp_path / "waiting.json"
    missing = tmp_path / "missing.json"
    _gate(passing)
    _gate(waiting, passing=False)

    report = preflight_sources(
        {"passing": passing, "waiting": waiting, "missing": missing}
    )

    assert report["status"] == "WAIT_SOURCES"
    assert report["pass"] is False
    assert {row["code"] for row in report["blockers"]} == {
        "SOURCE_NOT_PASSING",
        "MISSING_SOURCE",
    }


def test_stage_plan_orders_cache_slack_migration_replay_and_figure() -> None:
    names = [stage.name for stage in build_stages(python="python-test")]

    assert names == [
        "merge_all_hardware_empty_cache",
        "certify_all_hardware_statewise_slack",
        "merge_loaded_action_ledger",
        "recalculate_migration_rates",
        "certify_migration_actions",
        "replay_hardware_quadrants",
        "export_paper_results",
        "render_quadrant_figure",
    ]
    assert all(stage.argv[0] == "python-test" for stage in build_stages(python="python-test"))


def test_pipeline_stops_at_first_failed_stage(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _gate(source)
    first_output = tmp_path / "first.json"
    failed_output = tmp_path / "failed.json"
    never_output = tmp_path / "never.json"
    stages = (
        Stage("first", ("first",), (first_output,)),
        Stage("failed", ("failed",), (failed_output,)),
        Stage("never", ("never",), (never_output,)),
    )
    calls = []

    def runner(argv):
        calls.append(tuple(argv))
        if argv[0] == "first":
            first_output.write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "ok", "")
        return subprocess.CompletedProcess(argv, 3, "waiting", "")

    report = run_pipeline(
        sources={"source": source},
        stages=stages,
        runner=runner,
    )

    assert report["status"] == "WAIT_STAGE"
    assert report["failed_stage"] == "failed"
    assert calls == [("first",), ("failed",)]
    assert not never_output.exists()


def test_pipeline_passes_only_when_every_declared_output_exists(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "derived.json"
    _gate(source)

    def runner(argv):
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    report = run_pipeline(
        sources={"source": source},
        stages=(Stage("only", ("only",), (output,)),),
        runner=runner,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["stages"][0]["outputs"][0]["sha256"]
