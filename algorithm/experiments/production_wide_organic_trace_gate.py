"""Production-wide organic trace boundary gate.

This gate is the reviewer-facing wrapper for the strongest production-trace
claim.  It intentionally separates three different notions that are easy to
conflate:

* queued-production live theorem trace readiness;
* organic canary recorder readiness;
* large-scale organic launched-completion evidence.

The strong production-wide claim is true when either a real organic live trace
has enough admitted launches/completions or the strict scheduler-history
completion certificate closes.  Controlled benchmark traces, dry-run oracle
traces, and active-progress shadow traces remain useful scoped evidence, but
they do not close this gate by themselves.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .organic_production_canary_recorder_gate import (
    DEFAULT_TRACE_PATH as DEFAULT_ORGANIC_TRACE,
    build_organic_production_canary_recorder_gate,
)
from .production_launch_completion_gate import build_production_launch_completion_gate
from .production_live_theorem_trace_gate import build_production_live_theorem_trace_gate
from .production_load_certificate import load_scheduler_records, _file_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_production_wide_organic_trace_gate(
    *,
    trace_path: str | Path = DEFAULT_ORGANIC_TRACE,
    min_trace_slots: int = 32,
    allow_launch: bool = False,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = state_dir / "queue.json"
    archive = state_dir / "queue_archive.jsonl"
    return copy.deepcopy(_cached_production_wide_organic_trace_gate(
        str(trace),
        *_file_fingerprint(trace),
        int(min_trace_slots),
        bool(allow_launch),
        str(queue),
        *_file_fingerprint(queue),
        str(archive),
        *_file_fingerprint(archive),
    ))


@lru_cache(maxsize=8)
def _cached_production_wide_organic_trace_gate(
    trace_path: str,
    trace_mtime_ns: int,
    trace_size: int,
    min_trace_slots: int,
    allow_launch: bool,
    queue_path: str,
    queue_mtime_ns: int,
    queue_size: int,
    archive_path: str,
    archive_mtime_ns: int,
    archive_size: int,
) -> dict[str, Any]:
    records = load_scheduler_records(copy_records=False)
    live_trace = build_production_live_theorem_trace_gate(
        records=records,
        trace_path=trace_path,
        min_trace_slots=min_trace_slots,
    )
    canary = build_organic_production_canary_recorder_gate(trace_path=trace_path)
    launch = build_production_launch_completion_gate(allow_launch=allow_launch)

    recorder_ready = bool(canary.get("organic_production_canary_recorder_ready"))
    queued_trace_ready_or_wait = bool(live_trace.get("pass"))
    active_progress_ready = bool(launch.get("large_scale_active_progress_ready"))
    trace_completion_ready = bool(
        canary.get("live_trace_large_scale_organic_completion_ready")
        or canary.get("trace_large_scale_organic_launched_completion_ready")
    )
    history_completion_ready = bool(
        canary.get("history_large_scale_organic_completion_ready")
        or canary.get("history_large_scale_organic_launched_completion_ready")
    )
    combined_completion_ready = bool(trace_completion_ready or history_completion_ready)
    live_trace_closed = bool(live_trace.get("production_wide_live_trace_closed"))
    strong_ready = bool(
        (live_trace_closed and trace_completion_ready)
        or history_completion_ready
    ) and not canary.get("unadmitted_launched_count")
    scoped_ready = recorder_ready

    return {
        "gate": "production_wide_organic_trace_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "PRODUCTION_WIDE_HISTORY_COMPLETION_PASS_LIVE_TRACE_FALSE"
            if history_completion_ready and not (live_trace_closed and trace_completion_ready) else
            "PRODUCTION_WIDE_LIVE_TRACE_COMPLETION_PASS"
            if strong_ready else "PRODUCTION_WIDE_ORGANIC_TRACE_GATE_READY_STRONG_PENDING"
        ),
        "gate_pass": scoped_ready,
        "scoped_claim_ready": scoped_ready,
        "strong_claim_ready": strong_ready,
        "production_wide_live_trace_closed": live_trace_closed,
        "production_wide_history_completion_closed": history_completion_ready,
        "history_large_scale_organic_completion_ready": history_completion_ready,
        "live_trace_large_scale_organic_completion_ready": trace_completion_ready,
        "combined_large_scale_completion_evidence_ready": combined_completion_ready,
        "organic_production_canary_recorder_ready": recorder_ready,
        "queued_live_trace_gate_ready_or_wait": queued_trace_ready_or_wait,
        "large_scale_active_progress_ready": active_progress_ready,
        "large_scale_organic_launched_completion_ready": combined_completion_ready,
        "large_scale_organic_launched_completion_ready_semantics": (
            "legacy combined alias; use history_large_scale_organic_completion_ready, "
            "live_trace_large_scale_organic_completion_ready, and "
            "combined_large_scale_completion_evidence_ready"
        ),
        "trace_large_scale_organic_launched_completion_ready": trace_completion_ready,
        "history_large_scale_organic_launched_completion_ready": history_completion_ready,
        "safe_launch_gate_status": launch.get("launch_status"),
        "allow_launch": bool(allow_launch),
        "live_trace_snapshot": {
            "status": live_trace.get("status"),
            "production_wide_live_trace_closed": live_trace.get("production_wide_live_trace_closed"),
            "production_queued_count": (live_trace.get("queue_summary") or {}).get("production_queued_count"),
            "admissible_production_queued_count": (
                live_trace.get("queue_summary") or {}
            ).get("admissible_production_queued_count"),
            "trace_slot_count": (live_trace.get("trace_report") or {}).get("slot_count"),
            "trace_theorem_slot_count": (live_trace.get("trace_report") or {}).get("theorem_slot_count"),
        },
        "canary_snapshot": {
            "trace_exists": canary.get("trace_exists"),
            "trace_row_count": canary.get("trace_row_count"),
            "theorem_trace_row_count": canary.get("theorem_trace_row_count"),
            "organic_production_launched_count": canary.get("organic_production_launched_count"),
            "organic_production_completed_count": canary.get("organic_production_completed_count"),
            "organic_production_completion_fraction": canary.get("organic_production_completion_fraction"),
            "workload_domain_count": canary.get("workload_domain_count"),
            "node_count": canary.get("node_count"),
            "unadmitted_launched_count": canary.get("unadmitted_launched_count"),
            "trace_large_scale_organic_launched_completion_ready": canary.get(
                "trace_large_scale_organic_launched_completion_ready"
            ),
            "live_trace_large_scale_organic_completion_ready": canary.get(
                "live_trace_large_scale_organic_completion_ready"
            ),
            "history_large_scale_organic_launched_completion_ready": canary.get(
                "history_large_scale_organic_launched_completion_ready"
            ),
            "history_large_scale_organic_completion_ready": canary.get(
                "history_large_scale_organic_completion_ready"
            ),
            "combined_large_scale_completion_evidence_ready": canary.get(
                "combined_large_scale_completion_evidence_ready"
            ),
            "history_completion_snapshot": canary.get("history_completion_snapshot"),
            "next_threshold": canary.get("next_threshold"),
        },
        "launch_snapshot": {
            "status": launch.get("status"),
            "launch_status": launch.get("launch_status"),
            "launch_safe": launch.get("launch_safe"),
            "active_production_count": launch.get("active_production_count"),
            "queued_production_count": launch.get("queued_production_count"),
            "running_production_count": launch.get("running_production_count"),
            "progress_observation_count": launch.get("progress_observation_count"),
            "large_scale_active_progress_ready": launch.get("large_scale_active_progress_ready"),
            "large_scale_launched_completion_ready": launch.get("large_scale_launched_completion_ready"),
        },
        "blocker": (
            "" if strong_ready else
            "Strong production-wide organic launched-completion evidence requires "
            "either a naturally accumulated admitted live oracle trace or the "
            "strict scheduler-history completion certificate.  Controlled "
            "ScheduleurmBench traces and shadow traces are excluded."
        ),
        "scope": (
            "Executable boundary for production-wide organic trace claims.  The "
            "scoped gate passes when recorder/admission machinery is ready.  "
            "Queued live-trace closure, strict scheduler-history completion, "
            "and organic launched completion are reported separately.  The "
            "history path closes completed production evidence without claiming "
            "that every live slot has an emitted oracle trace row."
        ),
        "pass": scoped_ready,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    live = report.get("live_trace_snapshot") or {}
    canary = report.get("canary_snapshot") or {}
    launch = report.get("launch_snapshot") or {}
    lines = [
        "# Production-Wide Organic Trace Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `production_wide_live_trace_closed` | {str(bool(report.get('production_wide_live_trace_closed'))).lower()} |",
        f"| `production_wide_history_completion_closed` | {str(bool(report.get('production_wide_history_completion_closed'))).lower()} |",
        f"| `history_large_scale_organic_completion_ready` | {str(bool(report.get('history_large_scale_organic_completion_ready'))).lower()} |",
        f"| `live_trace_large_scale_organic_completion_ready` | {str(bool(report.get('live_trace_large_scale_organic_completion_ready'))).lower()} |",
        f"| `combined_large_scale_completion_evidence_ready` | {str(bool(report.get('combined_large_scale_completion_evidence_ready'))).lower()} |",
        f"| `organic_production_canary_recorder_ready` | {str(bool(report.get('organic_production_canary_recorder_ready'))).lower()} |",
        f"| `queued_live_trace_gate_ready_or_wait` | {str(bool(report.get('queued_live_trace_gate_ready_or_wait'))).lower()} |",
        f"| `large_scale_active_progress_ready` | {str(bool(report.get('large_scale_active_progress_ready'))).lower()} |",
        f"| `trace_large_scale_organic_launched_completion_ready` | {str(bool(report.get('trace_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `history_large_scale_organic_launched_completion_ready` | {str(bool(report.get('history_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `safe_launch_gate_status` | `{report.get('safe_launch_gate_status')}` |",
        f"| `production_queued_count` | {live.get('production_queued_count')} |",
        f"| `admissible_production_queued_count` | {live.get('admissible_production_queued_count')} |",
        f"| `trace_theorem_slot_count` | {live.get('trace_theorem_slot_count')} |",
        f"| `organic_launches` | {canary.get('organic_production_launched_count')} |",
        f"| `organic_completions` | {canary.get('organic_production_completed_count')} |",
        f"| `active_progress_observations` | {launch.get('progress_observation_count')} |",
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    return "\n".join(lines)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_production_wide_organic_trace_gate(
        trace_path=args.trace_path,
        min_trace_slots=args.min_trace_slots,
        allow_launch=args.allow_launch,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.production_wide_organic_trace_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build production-wide organic trace boundary gate")
    build.add_argument("--trace-path", default=str(DEFAULT_ORGANIC_TRACE))
    build.add_argument("--min-trace-slots", type=int, default=32)
    build.add_argument("--allow-launch", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "production_wide_organic_trace_gate_20260614.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "production_wide_organic_trace_gate_20260614.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
