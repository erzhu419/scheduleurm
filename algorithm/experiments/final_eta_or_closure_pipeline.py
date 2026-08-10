"""Run the deterministic post-measurement OR closure pipeline.

This module never launches or migrates jobs.  It starts only after all frozen
natural-completion gates pass, then rebuilds every downstream cache,
certificate, replay, and figure in dependency order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from algorithm.experiments.critical_gpu_all_hardware_lcb_cache_merge import (
    DEFAULT_JTL311_GATE,
)
from algorithm.experiments.critical_gpu_available_lcb_cache_merge import (
    DEFAULT_REPORT as AVAILABLE_REPORT,
)
from algorithm.experiments.critical_gpu_loaded_action_ledger_merge import (
    default_gate_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "final_eta_or_closure_pipeline_20260810.json"
FIGURE_SCRIPT = REPO_ROOT / "md" / "figures" / "make_scheduleurm_quadrant_sota_grid.py"
FIGURE_STEM = REPO_ROOT / "md" / "figures" / "scheduleurm_quadrant_sota_grid"


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def required_sources() -> dict[str, Path]:
    sources = {
        "available_empty_cache_report": Path(AVAILABLE_REPORT),
        "jtl311linux_empty_lcb_gate": Path(DEFAULT_JTL311_GATE),
    }
    sources.update(
        {
            f"{node}_loaded_lcb_gate": path
            for node, path in default_gate_paths().items()
        }
    )
    return sources


def preflight_sources(sources: Mapping[str, Path] | None = None) -> dict[str, Any]:
    supplied = dict(sources or required_sources())
    rows = []
    blockers = []
    for label, raw_path in supplied.items():
        path = Path(raw_path).resolve()
        row: dict[str, Any] = {"label": label, "path": str(path)}
        if not path.is_file():
            row["status"] = "MISSING"
            blockers.append({"code": "MISSING_SOURCE", "label": label, "path": str(path)})
            rows.append(row)
            continue
        raw = path.read_bytes()
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            row["status"] = "INVALID_JSON"
            blockers.append({"code": "INVALID_JSON", "label": label, "detail": str(exc)})
            rows.append(row)
            continue
        row["gate"] = payload.get("gate")
        row["gate_status"] = payload.get("status")
        row["gate_pass"] = payload.get("pass")
        if payload.get("status") != "PASS" or payload.get("pass") is not True:
            row["status"] = "NOT_PASSING"
            blockers.append(
                {
                    "code": "SOURCE_NOT_PASSING",
                    "label": label,
                    "gate_status": payload.get("status"),
                    "gate_pass": payload.get("pass"),
                }
            )
        else:
            row["status"] = "PASS"
        rows.append(row)
    return {
        "status": "PASS" if not blockers else "WAIT_SOURCES",
        "pass": not blockers,
        "source_rows": rows,
        "blockers": blockers,
    }


def build_stages(*, python: str = sys.executable, include_figure: bool = True) -> tuple[Stage, ...]:
    stages = [
        Stage(
            "merge_all_hardware_empty_cache",
            (python, "-m", "algorithm.experiments.critical_gpu_all_hardware_lcb_cache_merge"),
            (
                ARTIFACT_ROOT / "critical_gpu_all_hardware_lcb_cache_merge_20260809.json",
                ARTIFACT_ROOT / "service_cache_v2_all_hardware_gpu_phase_20260809.json",
            ),
        ),
        Stage(
            "certify_all_hardware_statewise_slack",
            (python, "-m", "algorithm.experiments.critical_gpu_all_hardware_statewise_slack_certificate"),
            (
                ARTIFACT_ROOT / "critical_gpu_all_hardware_statewise_slack_20260810.json",
                REPO_ROOT / "md" / "critical_gpu_all_hardware_statewise_slack_20260810.md",
            ),
        ),
        Stage(
            "merge_loaded_action_ledger",
            (python, "-m", "algorithm.experiments.critical_gpu_loaded_action_ledger_merge"),
            (
                ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_merge_20260809.json",
                ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_20260809.json",
                ARTIFACT_ROOT / "service_cache_v2_cpu_gpu_loaded_final_20260809.json",
            ),
        ),
        Stage(
            "recalculate_migration_rates",
            (python, "-m", "algorithm.experiments.migration_rate_recalibration_gate"),
            (
                ARTIFACT_ROOT / "migration_rate_recalibration_gate_20260809.json",
                ARTIFACT_ROOT / "migration_rate_recalibration_gate_20260809.md",
            ),
        ),
        Stage(
            "certify_migration_actions",
            (python, "-m", "algorithm.experiments.unified_migration_recalibration_certificate"),
            (
                ARTIFACT_ROOT / "unified_migration_recalibration_certificate_20260809.json",
                ARTIFACT_ROOT / "unified_migration_recalibration_certificate_20260809.md",
            ),
        ),
        Stage(
            "replay_hardware_quadrants",
            (python, "-m", "algorithm.experiments.unified_hardware_or_replay"),
            (ARTIFACT_ROOT / "unified_hardware_or_replay_20260809.json",),
        ),
        Stage(
            "export_paper_results",
            (python, "-m", "algorithm.experiments.unified_hardware_paper_results"),
            (
                ARTIFACT_ROOT / "unified_hardware_paper_results_20260810.json",
                REPO_ROOT / "md" / "unified_hardware_paper_results_20260810.md",
                REPO_ROOT / "paper" / "generated" / "unified_hardware_results.tex",
            ),
        ),
    ]
    if include_figure:
        stages.append(
            Stage(
                "render_quadrant_figure",
                (python, str(FIGURE_SCRIPT)),
                tuple(
                    Path(f"{FIGURE_STEM}.{suffix}")
                    for suffix in ("svg", "pdf", "png", "tiff")
                )
                + (FIGURE_STEM.with_name(FIGURE_STEM.name + "_data.json"),),
            )
        )
    return tuple(stages)


def run_pipeline(
    *,
    sources: Mapping[str, Path] | None = None,
    stages: Sequence[Stage] | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    started = time.time()
    preflight = preflight_sources(sources)
    report: dict[str, Any] = {
        "gate": "final_eta_or_closure_pipeline",
        "schema_version": 1,
        "status": preflight["status"],
        "pass": False,
        "started_at": started,
        "launches_remote_work": False,
        "touches_running_tasks": False,
        "preflight": preflight,
        "stages": [],
    }
    if preflight["pass"] is not True:
        report["finished_at"] = time.time()
        return report

    execute = runner or _run
    for stage in tuple(stages or build_stages()):
        row: dict[str, Any] = {
            "name": stage.name,
            "argv": list(stage.argv),
            "started_at": time.time(),
        }
        completed = execute(stage.argv)
        row.update(
            {
                "finished_at": time.time(),
                "returncode": int(completed.returncode),
                "stdout_tail": (completed.stdout or "")[-4000:],
                "stderr_tail": (completed.stderr or "")[-4000:],
            }
        )
        output_rows = []
        for output in stage.outputs:
            path = Path(output).resolve()
            output_row: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if path.is_file():
                output_row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            output_rows.append(output_row)
        row["outputs"] = output_rows
        row["pass"] = completed.returncode == 0 and all(item["exists"] for item in output_rows)
        report["stages"].append(row)
        if not row["pass"]:
            report["status"] = "WAIT_STAGE" if completed.returncode == 3 else "FAIL_STAGE"
            report["failed_stage"] = stage.name
            report["finished_at"] = time.time()
            return report

    report["status"] = "PASS"
    report["pass"] = True
    report["finished_at"] = time.time()
    return report


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.preflight_only:
        preflight = preflight_sources()
        report = {
            "gate": "final_eta_or_closure_pipeline",
            "schema_version": 1,
            "status": preflight["status"],
            "pass": preflight["pass"],
            "preflight_only": True,
            "preflight": preflight,
        }
    else:
        report = run_pipeline(stages=build_stages(include_figure=not args.skip_figure))
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
