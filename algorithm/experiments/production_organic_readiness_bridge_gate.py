"""Production organic readiness bridge.

When the rolling production queue has no admissible queued work, a live launch
claim still cannot be manufactured.  This gate therefore separates live arrival
readiness, queued theorem traces, non-invasive active-production theorem
shadow traces, and the strict scheduler-history completion certificate.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .organic_production_canary_recorder_gate import build_organic_production_canary_recorder_gate
from .production_live_theorem_trace_gate import build_production_live_theorem_trace_gate
from .production_load_certificate import load_scheduler_records, _file_fingerprint
from .production_queued_theorem_trace import generate_production_queued_theorem_trace
from .production_shadow_theorem_trace import build_production_shadow_theorem_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "production_organic_readiness_bridge_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "production_organic_readiness_bridge_gate_20260614.md"
DEFAULT_TRACE = ARTIFACT_ROOT / "production_organic_readiness_shadow_trace_20260614.jsonl"
DEFAULT_QUEUED_TRACE = ARTIFACT_ROOT / "production_organic_readiness_queued_trace_20260614.jsonl"


def build_production_organic_readiness_bridge_gate(
    *,
    trace_path: str | Path = DEFAULT_TRACE,
    queued_trace_path: str | Path = DEFAULT_QUEUED_TRACE,
    min_shadow_tasks: int = 8,
    max_shadow_tasks: int = 96,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    queued_trace = Path(queued_trace_path).expanduser()
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = state_dir / "queue.json"
    archive = state_dir / "queue_archive.jsonl"
    return copy.deepcopy(_cached_production_organic_readiness_bridge_gate(
        str(trace),
        str(queued_trace),
        int(min_shadow_tasks),
        int(max_shadow_tasks),
        str(queue),
        *_file_fingerprint(queue),
        str(archive),
        *_file_fingerprint(archive),
    ))


@lru_cache(maxsize=8)
def _cached_production_organic_readiness_bridge_gate(
    trace_path: str,
    queued_trace_path: str,
    min_shadow_tasks: int,
    max_shadow_tasks: int,
    queue_path: str,
    queue_mtime_ns: int,
    queue_size: int,
    archive_path: str,
    archive_mtime_ns: int,
    archive_size: int,
) -> dict[str, Any]:
    trace_path = Path(trace_path)
    queued_trace_path = Path(queued_trace_path)
    records = load_scheduler_records(copy_records=False)
    live = build_production_live_theorem_trace_gate(records=records, min_trace_slots=32)
    canary = build_organic_production_canary_recorder_gate()
    if int(max_shadow_tasks) <= 0:
        shadow = {
            "production_shadow_trace_closed": False,
            "active_production_shadow_task_count": 0,
            "trace_slot_count": 0,
            "theorem_slot_count": 0,
            "candidate_count_total": 0,
        }
    else:
        shadow = build_production_shadow_theorem_trace(
            trace_path=trace_path,
            output_path=ARTIFACT_ROOT / "production_organic_readiness_shadow_trace_20260614.json",
            max_tasks=max_shadow_tasks,
            algorithm="theorem_maxweight_v1",
            hard_rule_mode="",
            uncertified_mode="block",
            reserve_shadow_capacity=True,
            append=False,
        )
    queued_trace = generate_production_queued_theorem_trace(
        trace_path=queued_trace_path,
        output_path=ARTIFACT_ROOT / "production_organic_readiness_queued_trace_20260614.json",
        max_tasks=64,
        max_tasks_per_gpu=8,
        max_post_vram_frac=0.95,
        algorithm="theorem_maxweight_v1",
        hard_rule_mode="clean_bench",
        allow_all_hard_rule_scope=True,
        append=False,
    )
    shadow_count = int(shadow.get("active_production_shadow_task_count") or 0)
    shadow_closed = bool(shadow.get("production_shadow_trace_closed"))
    arrival_wait = live.get("status") in {
        "WAIT_NO_QUEUED_PRODUCTION",
        "WAIT_NO_ADMISSIBLE_QUEUED_PRODUCTION",
    }
    queued_theorem_ready = bool(queued_trace.get("pass"))
    history_completion_ready = bool(
        canary.get("history_large_scale_organic_completion_ready")
        or canary.get("history_large_scale_organic_launched_completion_ready")
    )
    trace_completion_ready = bool(
        canary.get("live_trace_large_scale_organic_completion_ready")
        or canary.get("trace_large_scale_organic_launched_completion_ready")
    )
    combined_completion_ready = bool(history_completion_ready or trace_completion_ready)
    readiness_ready = bool(
        canary.get("organic_production_canary_recorder_ready")
        and shadow_closed
        and shadow_count >= int(min_shadow_tasks)
        and (bool(live.get("pass")) or queued_theorem_ready)
    )
    strong_ready = bool(
        (
            live.get("production_wide_live_trace_closed")
            and trace_completion_ready
        )
        or history_completion_ready
    )
    return {
        "gate": "production_organic_readiness_bridge_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "PRODUCTION_ORGANIC_STRONG_COMPLETION_CLOSED"
            if strong_ready else
            "PRODUCTION_ORGANIC_READINESS_CLOSED_QUEUED_THEOREM_TRACE_READY"
            if readiness_ready and queued_theorem_ready else
            "PRODUCTION_ORGANIC_READINESS_CLOSED_WAITING_FOR_ARRIVALS"
            if readiness_ready and arrival_wait else
            "PRODUCTION_ORGANIC_READINESS_PENDING"
        ),
        "organic_readiness_bridge_ready": bool(readiness_ready),
        "arrival_population_wait": bool(arrival_wait),
        "live_trace_status": live.get("status"),
        "production_queued_count": (live.get("queue_summary") or {}).get("production_queued_count"),
        "admissible_production_queued_count": (
            live.get("queue_summary") or {}
        ).get("admissible_production_queued_count"),
        "canary_recorder_ready": bool(canary.get("organic_production_canary_recorder_ready")),
        "queued_theorem_trace_ready": bool(queued_theorem_ready),
        "queued_theorem_trace_task_count": int(queued_trace.get("queued_admissible_count") or 0),
        "queued_theorem_trace_slot_count": int(queued_trace.get("theorem_slot_count") or 0),
        "queued_theorem_trace_candidate_count_total": int(queued_trace.get("candidate_count_total") or 0),
        "queued_theorem_trace_alpha0": (queued_trace.get("oracle_bridge") or {}).get("alpha0"),
        "queued_theorem_trace_alpha1": (queued_trace.get("oracle_bridge") or {}).get("alpha1"),
        "queued_theorem_trace_hard_rule_mode": queued_trace.get("hard_rule_mode"),
        "queued_theorem_trace_launch_claim": False,
        "shadow_trace_closed": bool(shadow_closed),
        "shadow_task_count": shadow_count,
        "shadow_theorem_slot_count": int(shadow.get("theorem_slot_count") or 0),
        "shadow_candidate_count_total": int(shadow.get("candidate_count_total") or 0),
        "shadow_alpha0": (shadow.get("theorem_subset_oracle_bridge") or {}).get("alpha0"),
        "shadow_alpha1": (shadow.get("theorem_subset_oracle_bridge") or {}).get("alpha1"),
        "large_scale_organic_launched_completion_ready": bool(
            combined_completion_ready
        ),
        "large_scale_organic_launched_completion_ready_semantics": (
            "legacy combined alias; use history_large_scale_organic_completion_ready, "
            "live_trace_large_scale_organic_completion_ready, and "
            "combined_large_scale_completion_evidence_ready"
        ),
        "trace_large_scale_organic_launched_completion_ready": bool(
            trace_completion_ready
        ),
        "live_trace_large_scale_organic_completion_ready": bool(trace_completion_ready),
        "history_large_scale_organic_launched_completion_ready": bool(history_completion_ready),
        "history_large_scale_organic_completion_ready": bool(history_completion_ready),
        "combined_large_scale_completion_evidence_ready": bool(combined_completion_ready),
        "history_completion_snapshot": canary.get("history_completion_snapshot"),
        "production_wide_live_trace_closed": bool(live.get("production_wide_live_trace_closed")),
        "strong_claim_ready": bool(strong_ready),
        "scoped_claim_ready": bool(readiness_ready),
        "pass": bool(readiness_ready),
        "blocker": "" if readiness_ready else _blocker(live, canary, shadow, queued_trace, min_shadow_tasks),
        "scope": (
            "This gate closes the production organic readiness bridge: strict "
            "admission, canary recorder, and active-production theorem shadow "
            "semantics.  If queued production exists, it also builds a read-only "
            "theorem trace for those queued rows under the opt-in clean-bench "
            "hard-rule mode.  It does not launch work or manufacture completed "
            "organic production rows.  Large-scale completion can also close via "
            "the strict scheduler-history completion certificate."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Production Organic Readiness Bridge Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `organic_readiness_bridge_ready` | {str(bool(report.get('organic_readiness_bridge_ready'))).lower()} |",
        f"| `arrival_population_wait` | {str(bool(report.get('arrival_population_wait'))).lower()} |",
        f"| `live_trace_status` | `{report.get('live_trace_status')}` |",
        f"| `production_queued_count` | {report.get('production_queued_count')} |",
        f"| `admissible_production_queued_count` | {report.get('admissible_production_queued_count')} |",
        f"| `canary_recorder_ready` | {str(bool(report.get('canary_recorder_ready'))).lower()} |",
        f"| `queued_theorem_trace_ready` | {str(bool(report.get('queued_theorem_trace_ready'))).lower()} |",
        f"| `queued_theorem_trace_task_count` | {report.get('queued_theorem_trace_task_count')} |",
        f"| `queued_theorem_trace_slot_count` | {report.get('queued_theorem_trace_slot_count')} |",
        f"| `shadow_trace_closed` | {str(bool(report.get('shadow_trace_closed'))).lower()} |",
        f"| `shadow_task_count` | {report.get('shadow_task_count')} |",
        f"| `shadow_theorem_slot_count` | {report.get('shadow_theorem_slot_count')} |",
        f"| `large_scale_organic_launched_completion_ready` | {str(bool(report.get('large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `trace_large_scale_organic_launched_completion_ready` | {str(bool(report.get('trace_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `live_trace_large_scale_organic_completion_ready` | {str(bool(report.get('live_trace_large_scale_organic_completion_ready'))).lower()} |",
        f"| `history_large_scale_organic_launched_completion_ready` | {str(bool(report.get('history_large_scale_organic_launched_completion_ready'))).lower()} |",
        f"| `history_large_scale_organic_completion_ready` | {str(bool(report.get('history_large_scale_organic_completion_ready'))).lower()} |",
        f"| `combined_large_scale_completion_evidence_ready` | {str(bool(report.get('combined_large_scale_completion_evidence_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
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


def _blocker(
    live: Mapping[str, Any],
    canary: Mapping[str, Any],
    shadow: Mapping[str, Any],
    queued_trace: Mapping[str, Any],
    min_shadow_tasks: int,
) -> str:
    if not canary.get("organic_production_canary_recorder_ready"):
        return "organic canary recorder/admission contract is not ready"
    if not shadow.get("production_shadow_trace_closed"):
        return "active-production theorem shadow trace is not closed"
    if int(shadow.get("active_production_shadow_task_count") or 0) < int(min_shadow_tasks):
        return f"fewer than {int(min_shadow_tasks)} active production tasks for the readiness shadow"
    if not live.get("pass"):
        if queued_trace.get("pass"):
            return ""
        return str(live.get("status") or "production live theorem gate did not pass or wait safely")
    if not queued_trace.get("pass") and (live.get("queue_summary") or {}).get("production_queued_count"):
        return "queued production exists but the read-only theorem queued trace is not closed"
    return "readiness condition failed for an unclassified reason"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_production_organic_readiness_bridge_gate(
        trace_path=args.trace_path,
        queued_trace_path=args.queued_trace_path,
        min_shadow_tasks=args.min_shadow_tasks,
        max_shadow_tasks=args.max_shadow_tasks,
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
        prog="python -m algorithm.experiments.production_organic_readiness_bridge_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build production organic readiness bridge")
    build.add_argument("--trace-path", default=str(DEFAULT_TRACE))
    build.add_argument("--queued-trace-path", default=str(DEFAULT_QUEUED_TRACE))
    build.add_argument("--min-shadow-tasks", type=int, default=8)
    build.add_argument("--max-shadow-tasks", type=int, default=96)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
