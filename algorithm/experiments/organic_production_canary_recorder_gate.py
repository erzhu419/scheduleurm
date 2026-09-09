"""Organic production canary recorder gate.

The canary recorder is the strict bridge between future production arrivals and
the theorem-facing population.  It is intentionally non-invasive by default:
it audits admission and trace readiness, reads the rolling production
launch/completion gate, and promotes launched-completion claims only when
either live trace thresholds or strict scheduler-history thresholds are met.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .future_production_admission_contract import build_future_production_admission_contract
from .organic_history_completion_gate import build_organic_history_completion_gate
from .production_launch_completion_gate import build_production_launch_completion_gate
from .production_load_certificate import load_scheduler_records, _file_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRACE_PATH = Path.home() / ".claude" / "scheduler" / "theorem_oracle_trace.jsonl"


def build_organic_production_canary_recorder_gate(
    *,
    trace_path: str | Path | None = None,
    launch_threshold: int = 64,
    completion_threshold: int = 50,
    completion_fraction_threshold: float = 0.80,
    workload_domain_threshold: int = 3,
    node_threshold: int = 2,
    use_history_completion: bool = True,
) -> dict[str, Any]:
    trace_path = Path(trace_path or os.environ.get("SCHEDULEURM_ORACLE_TRACE_PATH") or DEFAULT_TRACE_PATH).expanduser()
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = state_dir / "queue.json"
    archive = state_dir / "queue_archive.jsonl"
    return copy.deepcopy(_cached_organic_production_canary_recorder_gate(
        str(trace_path),
        *_file_fingerprint(trace_path),
        int(launch_threshold),
        int(completion_threshold),
        float(completion_fraction_threshold),
        int(workload_domain_threshold),
        int(node_threshold),
        bool(use_history_completion),
        str(queue),
        *_file_fingerprint(queue),
        str(archive),
        *_file_fingerprint(archive),
    ))


@lru_cache(maxsize=16)
def _cached_organic_production_canary_recorder_gate(
    trace_path: str,
    trace_mtime_ns: int,
    trace_size: int,
    launch_threshold: int,
    completion_threshold: int,
    completion_fraction_threshold: float,
    workload_domain_threshold: int,
    node_threshold: int,
    use_history_completion: bool,
    queue_path: str,
    queue_mtime_ns: int,
    queue_size: int,
    archive_path: str,
    archive_mtime_ns: int,
    archive_size: int,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    admission = build_future_production_admission_contract(
        records=load_scheduler_records(copy_records=False),
        admission_mode="strict",
    )
    production = build_production_launch_completion_gate(allow_launch=False)
    trace_rows = _read_trace_rows(trace_path)
    launched_rows = [row for row in trace_rows if _event(row) in {"dispatch", "launch", "selected_action"}]
    completed_rows = [row for row in trace_rows if _event(row) in {"completion", "completed", "done"}]
    theorem_rows = [
        row for row in trace_rows
        if row.get("score_semantics") == "robust_maxweight_lower_service"
        or bool(row.get("theorem_semantics_available"))
    ]
    unadmitted_rows = [
        row for row in launched_rows
        if str(row.get("theorem_admission_route") or "ADMIT_THEOREM_TRACE") == "PROBE_REQUIRED"
    ]
    domains = {
        str(row.get("theorem_workload_key") or row.get("workload_key") or row.get("class_key") or "")
        for row in theorem_rows
    }
    domains.discard("")
    nodes = {
        str(row.get("node") or row.get("selected_node") or "")
        for row in theorem_rows
    }
    nodes.discard("")
    launched_count = len(launched_rows)
    completed_count = len(completed_rows)
    completion_fraction = float(completed_count) / float(launched_count) if launched_count else 0.0
    trace_completion_ready = (
        launched_count >= int(launch_threshold)
        and completed_count >= int(completion_threshold)
        and completion_fraction >= float(completion_fraction_threshold)
        and len(domains) >= int(workload_domain_threshold)
        and len(nodes) >= int(node_threshold)
        and not unadmitted_rows
        and theorem_rows
    )
    history_completion = (
        build_organic_history_completion_gate(
            launch_threshold=launch_threshold,
            completion_threshold=completion_threshold,
            completion_fraction_threshold=completion_fraction_threshold,
            workload_domain_threshold=workload_domain_threshold,
            node_threshold=node_threshold,
        )
        if use_history_completion else {}
    )
    history_completion_ready = bool(
        history_completion.get("history_large_scale_organic_completion_ready")
        or history_completion.get("large_scale_organic_history_completion_ready")
    )
    organic_ready = bool(trace_completion_ready or history_completion_ready)
    recorder_ready = (
        bool(admission.get("theorem_admission_default_trace_mode_ready"))
        and bool(admission.get("strict_theorem_admission_ready", True))
    )
    return {
        "gate": "organic_production_canary_recorder_gate",
        "status": (
            "ORGANIC_PRODUCTION_CANARY_COMPLETION_PASS"
            if trace_completion_ready else
            "ORGANIC_PRODUCTION_CANARY_HISTORY_COMPLETION_PASS"
            if history_completion_ready else
            "ORGANIC_PRODUCTION_CANARY_RECORDER_READY_COMPLETION_PENDING"
        ),
        "gate_pass": recorder_ready,
        "scoped_claim_ready": recorder_ready,
        "strong_claim_ready": bool(trace_completion_ready),
        "combined_large_scale_completion_evidence_ready": organic_ready,
        "pass_meaning": (
            "strict organic production canary recorder readiness, not organic "
            "live-trace launched-completion closure unless strong_claim_ready "
            "is true.  History completion is reported separately through "
            "history_large_scale_organic_completion_ready and "
            "combined_large_scale_completion_evidence_ready"
        ),
        "trace_path": str(trace_path),
        "trace_exists": trace_path.exists(),
        "trace_row_count": len(trace_rows),
        "theorem_trace_row_count": len(theorem_rows),
        "organic_production_launched_count": launched_count,
        "organic_production_completed_count": completed_count,
        "organic_production_completion_fraction": completion_fraction,
        "workload_domain_count": len(domains),
        "node_count": len(nodes),
        "unadmitted_launched_count": len(unadmitted_rows),
        "launch_threshold": int(launch_threshold),
        "completion_threshold": int(completion_threshold),
        "completion_fraction_threshold": float(completion_fraction_threshold),
        "workload_domain_threshold": int(workload_domain_threshold),
        "node_threshold": int(node_threshold),
        "organic_production_canary_recorder_ready": recorder_ready,
        "trace_large_scale_organic_launched_completion_ready": bool(trace_completion_ready),
        "live_trace_large_scale_organic_completion_ready": bool(trace_completion_ready),
        "history_large_scale_organic_launched_completion_ready": bool(history_completion_ready),
        "history_large_scale_organic_completion_ready": bool(history_completion_ready),
        "history_completion_snapshot": {
            "status": history_completion.get("status"),
            "strict_organic_launched_count": history_completion.get("strict_organic_launched_count"),
            "strict_organic_completed_count": history_completion.get("strict_organic_completed_count"),
            "strict_organic_completion_fraction": history_completion.get("strict_organic_completion_fraction"),
            "strict_unadmitted_launched_count": history_completion.get("strict_unadmitted_launched_count"),
            "workload_domain_count": history_completion.get("workload_domain_count"),
            "node_count": history_completion.get("node_count"),
            "admission_mode": history_completion.get("admission_mode"),
        },
        "organic_production_progress_ready": bool(
            production.get("progress_observation_count", 0)
            and production.get("active_production_count", 0)
        ),
        "large_scale_organic_launched_completion_ready": organic_ready,
        "large_scale_organic_launched_completion_ready_semantics": (
            "legacy combined alias; use "
            "history_large_scale_organic_completion_ready, "
            "live_trace_large_scale_organic_completion_ready, and "
            "combined_large_scale_completion_evidence_ready"
        ),
        "admission_contract_snapshot": {
            "future_production_automatic_theorem_closure_ready": admission.get(
                "future_production_automatic_theorem_closure_ready"
            ),
            "theorem_admission_default_trace_mode_ready": admission.get(
                "theorem_admission_default_trace_mode_ready"
            ),
            "admitted_traceable_count": admission.get("admitted_traceable_count"),
            "probe_required_count": admission.get("probe_required_count"),
            "future_jobs_all_theorem_grade_without_probe": admission.get(
                "future_jobs_all_theorem_grade_without_probe"
            ),
        },
        "production_gate_snapshot": {
            "status": production.get("status"),
            "active_production_count": production.get("active_production_count"),
            "queued_production_count": production.get("queued_production_count"),
            "running_production_count": production.get("running_production_count"),
            "progress_observation_count": production.get("progress_observation_count"),
            "large_scale_active_progress_ready": production.get("large_scale_active_progress_ready"),
            "large_scale_launched_completion_ready": production.get("large_scale_launched_completion_ready"),
        },
        "environment_contract": {
            "SCHEDULEURM_ORACLE_TRACE_PATH": str(trace_path),
            "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": "block or trace_only",
            "SCHEDULEURM_THEOREM_ADMISSION_MODE": "strict",
            "SCHEDULEURM_ALGORITHM": "theorem_maxweight_v1 when theorem dispatch is enabled",
        },
        "blocker": (
            "" if organic_ready else
            "Organic launched-completion thresholds require enough naturally queued "
            "admitted production work, theorem trace rows, and completed launches; "
            "the recorder contract is ready but the completion population is not yet closed."
        ),
        "next_threshold": (
            "Closed by strict scheduler-history completion certificate."
            if organic_ready else
            f"Accumulate >={int(launch_threshold)} organic launches, >={int(completion_threshold)} "
            f"organic completions, completion fraction >= {float(completion_fraction_threshold):.2f}, "
            f">={int(workload_domain_threshold)} workload domains, >={int(node_threshold)} nodes, "
            "and zero unadmitted launched rows."
        ),
        "pass": recorder_ready,
        "scope": (
            "Non-invasive organic production canary recorder.  It establishes the "
            "trace/admission contract and audits current rows.  Large-scale "
            "completion may be closed either by the live oracle trace counters "
            "or by the strict organic scheduler-history completion certificate. "
            "The two evidence paths are reported separately, so zero live-trace "
            "counters do not imply a live-trace completion claim."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Organic Production Canary Recorder Gate",
        "",
        "This gate passes recorder readiness only. It does not make the organic launched-completion claim unless the completion thresholds pass.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `gate_pass` | {str(bool(report.get('gate_pass'))).lower()} |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `trace_exists` | {str(bool(report.get('trace_exists'))).lower()} |",
        f"| `trace_row_count` | {report.get('trace_row_count')} |",
        f"| `theorem_trace_row_count` | {report.get('theorem_trace_row_count')} |",
        f"| `organic_production_launched_count` | {report.get('organic_production_launched_count')} |",
        f"| `organic_production_completed_count` | {report.get('organic_production_completed_count')} |",
        f"| `organic_production_completion_fraction` | {report.get('organic_production_completion_fraction')} |",
        f"| `workload_domain_count` | {report.get('workload_domain_count')} |",
        f"| `node_count` | {report.get('node_count')} |",
        f"| `unadmitted_launched_count` | {report.get('unadmitted_launched_count')} |",
        f"| `organic_production_canary_recorder_ready` | {str(bool(report.get('organic_production_canary_recorder_ready'))).lower()} |",
        f"| `trace_large_scale_organic_launched_completion_ready` | {str(bool(report.get('trace_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `live_trace_large_scale_organic_completion_ready` | {str(bool(report.get('live_trace_large_scale_organic_completion_ready'))).lower()} |",
        f"| `history_large_scale_organic_launched_completion_ready` | {str(bool(report.get('history_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `history_large_scale_organic_completion_ready` | {str(bool(report.get('history_large_scale_organic_completion_ready'))).lower()} |",
        f"| `combined_large_scale_completion_evidence_ready` | {str(bool(report.get('combined_large_scale_completion_evidence_ready'))).lower()} |",
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Next Threshold",
        "",
        str(report.get("next_threshold") or ""),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    return "\n".join(lines)


def _event(row: Mapping[str, Any]) -> str:
    return str(row.get("event") or row.get("event_type") or row.get("type") or "").lower()


def _read_trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_organic_production_canary_recorder_gate(
        trace_path=args.trace_path,
        use_history_completion=not args.no_history_completion,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.organic_production_canary_recorder_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build organic production canary recorder gate")
    build.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    build.add_argument("--no-history-completion", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "organic_production_canary_recorder_gate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "organic_production_canary_recorder_gate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
