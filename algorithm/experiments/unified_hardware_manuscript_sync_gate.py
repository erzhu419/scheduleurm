"""Fail closed when the paper, replay, and figure use different ETA evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_REPLAY = ARTIFACT_ROOT / "unified_hardware_or_replay_20260809.json"
DEFAULT_RESULTS = ARTIFACT_ROOT / "unified_hardware_paper_results_20260810.json"
DEFAULT_FIGURE_DATA = (
    REPO_ROOT / "md" / "figures" / "scheduleurm_quadrant_sota_grid_data.json"
)
DEFAULT_TEX = REPO_ROOT / "paper" / "generated" / "unified_hardware_results.tex"
DEFAULT_MANUSCRIPT = REPO_ROOT / "paper" / "main.tex"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "unified_hardware_manuscript_sync_gate_20260810.json"

REQUIRED_MANUSCRIPT_TOKENS = (
    r"\input{generated/unified_hardware_results.tex}",
    r"\UnifiedOnlineComparisonCount",
    r"\UnifiedOnlineLegacyMakespanGeoRatio",
    r"\UnifiedOnlineLegacyMeanFlowGeoRatio",
    r"\UnifiedStaticLegacyRows",
    r"\UnifiedOnlineArrivalFamilyRows",
    r"\UnifiedOnlineQuadrantRows",
    r"\UnifiedSotaPolicyRows",
    r"\UnifiedAblationRows",
    r"\UnifiedHardwareSlackRows",
    r"\UnifiedQueueMeanBacklogMean",
    r"\UnifiedQueueMaxBacklogMean",
    r"\UnifiedMigrationMakespanGeoRatio",
    r"\UnifiedMigrationMeanFlowGeoRatio",
    r"\UnifiedMinimumHardwareEta",
)


def build_manuscript_sync_report(
    *,
    replay_path: Path = DEFAULT_REPLAY,
    results_path: Path = DEFAULT_RESULTS,
    figure_data_path: Path = DEFAULT_FIGURE_DATA,
    tex_path: Path = DEFAULT_TEX,
    manuscript_path: Path = DEFAULT_MANUSCRIPT,
) -> dict[str, Any]:
    paths = {
        "replay": Path(replay_path).resolve(),
        "results": Path(results_path).resolve(),
        "figure_data": Path(figure_data_path).resolve(),
        "generated_tex": Path(tex_path).resolve(),
        "manuscript": Path(manuscript_path).resolve(),
    }
    report: dict[str, Any] = {
        "gate": "unified_hardware_manuscript_sync_gate",
        "schema_version": 1,
        "status": "WAIT_INPUTS",
        "pass": False,
        "launches_remote_work": False,
        "touches_running_tasks": False,
        "paths": {key: str(value) for key, value in paths.items()},
        "errors": [],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        report["wait_reasons"] = [{"code": "MISSING_INPUT", "paths": missing}]
        return report

    try:
        raw = {key: path.read_bytes() for key, path in paths.items()}
        replay = json.loads(raw["replay"].decode("utf-8"))
        results = json.loads(raw["results"].decode("utf-8"))
        figure = json.loads(raw["figure_data"].decode("utf-8"))
        tex = raw["generated_tex"].decode("utf-8")
        manuscript = raw["manuscript"].decode("utf-8")
        checks = _checks(
            replay=replay,
            results=results,
            figure=figure,
            tex=tex,
            manuscript=manuscript,
            replay_sha256=hashlib.sha256(raw["replay"]).hexdigest(),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        report.update(
            {
                "status": "FAIL_INPUT_CONTRACT",
                "errors": [
                    {"code": "INVALID_INPUT", "detail": f"{type(exc).__name__}: {exc}"}
                ],
            }
        )
        return report

    failed = [name for name, passed in checks.items() if not passed]
    report.update(
        {
            "status": "PASS" if not failed else "FAIL_SYNC",
            "pass": not failed,
            "checks": checks,
            "failed_checks": failed,
            "sha256": {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()},
            "claim_boundary": (
                "PASS certifies only that the manuscript text, generated tables, and "
                "four-quadrant figure are derived from the same passing schema-v2 "
                "measured-cache replay. It does not convert policy-semantics baselines "
                "into direct external-system executions."
            ),
        }
    )
    return report


def _checks(
    *,
    replay: Mapping[str, Any],
    results: Mapping[str, Any],
    figure: Mapping[str, Any],
    tex: str,
    manuscript: str,
    replay_sha256: str,
) -> dict[str, bool]:
    hardware_ids = {
        str(row.get("scenario_id") or "")
        for row in replay.get("scenarios") or []
        if row.get("scenario_kind") == "hardware_local"
    }
    figure_ids = {str(value) for value in figure.get("included_hardware_scenario_ids") or []}
    return {
        "replay_schema_v2_pass": bool(
            replay.get("gate") == "unified_hardware_or_replay"
            and replay.get("schema_version") == 2
            and replay.get("status") == "PASS"
            and replay.get("pass") is True
        ),
        "paper_results_pass": bool(
            results.get("gate") == "unified_hardware_paper_results"
            and results.get("status") == "PASS"
            and results.get("pass") is True
        ),
        "results_replay_hash_matches": str(results.get("replay_sha256") or "")
        == replay_sha256,
        "figure_replay_hash_matches": str(figure.get("source_artifact_sha256") or "")
        == replay_sha256,
        "figure_schema_matches": bool(
            figure.get("source_gate") == "unified_hardware_or_replay"
            and figure.get("source_schema_version") == 2
            and int(figure.get("source_run_count") or 0) == int(replay.get("run_count") or 0)
            and int(figure.get("source_policy_count") or 0)
            == int(replay.get("policy_count") or 0)
        ),
        "figure_hardware_population_matches": bool(hardware_ids and figure_ids == hardware_ids),
        "generated_tex_replay_hash_matches": (
            f"% Source replay SHA256: {replay_sha256}" in tex
        ),
        "manuscript_consumes_generated_results": all(
            token in manuscript for token in REQUIRED_MANUSCRIPT_TOKENS
        ),
        "comparison_scope_preserved": bool(
            replay.get("comparison_kind") == "same-cache_policy-semantics"
            and replay.get("full_stack_external_binary_comparison") is False
        ),
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = Path(handle.name)
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-data", type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_manuscript_sync_report(
        replay_path=args.replay,
        results_path=args.results,
        figure_data_path=args.figure_data,
        tex_path=args.tex,
        manuscript_path=args.manuscript,
    )
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
