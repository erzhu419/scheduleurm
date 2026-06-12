"""Bridge theorem trace slots to realized scheduler progress records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import load_trace_slots

from .progress_units import task_progress_observation


DEFAULT_QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"
DEFAULT_ARCHIVE = Path.home() / ".claude" / "scheduler" / "queue_archive.jsonl"


def build_realization_bridge(
    *,
    trace_path: str | Path,
    queue_path: str | Path = DEFAULT_QUEUE,
    archive_path: str | Path = DEFAULT_ARCHIVE,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    slots = load_trace_slots(trace) if trace.exists() else []
    records = _load_records(queue_path=queue_path, archive_path=archive_path)
    by_id = {}
    for row in records:
        task_id = str(row.get("id") or row.get("task_id") or "").strip()
        if task_id:
            by_id[task_id] = row

    rows = []
    for slot in slots:
        task_id = str(slot.get("task_id") or "").strip()
        task = by_id.get(task_id)
        rows.append(_slot_realization_row(slot, task))
    matched = [row for row in rows if row.get("task_record_found")]
    completed = [row for row in rows if row.get("terminal_status") == "done"]
    progress = [row for row in rows if row.get("progress_rate_per_s", 0.0) > 0.0]
    theorem_slots = [
        row for row in rows
        if row.get("score_semantics") == "robust_maxweight_lower_service"
    ]
    status = "NO_TRACE"
    if rows and not matched:
        status = "DRYRUN_NO_REALIZATION"
    elif rows and len(matched) < len(rows):
        status = "PARTIAL_REALIZATION"
    elif rows and len(completed) == len(rows):
        status = "LIVE_COMPLETION_REALIZATION_PASS"
    elif rows and progress:
        status = "LIVE_PROGRESS_REALIZATION_PASS"
    elif rows:
        status = "LIVE_RECORDS_NO_PROGRESS_YET"
    return {
        "gate": "live_trace_realization_bridge",
        "status": status,
        "trace_path": str(trace),
        "trace_slot_count": len(slots),
        "theorem_trace_slot_count": len(theorem_slots),
        "matched_task_record_count": len(matched),
        "completed_task_count": len(completed),
        "progress_observation_count": len(progress),
        "usable_for_live_completion_claim": status == "LIVE_COMPLETION_REALIZATION_PASS",
        "usable_for_live_progress_claim": status in {
            "LIVE_COMPLETION_REALIZATION_PASS",
            "LIVE_PROGRESS_REALIZATION_PASS",
        },
        "rows": rows,
        "scope": (
            "matches scheduler-emitted trace task ids to queue/archive records. "
            "Dry-run theorem traces intentionally have no realized completion "
            "claim until the same trace path is produced by actual dispatch."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live Trace Realization Bridge",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `status` | `{report.get('status')}` |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `theorem_trace_slot_count` | {report.get('theorem_trace_slot_count', 0)} |",
        f"| `matched_task_record_count` | {report.get('matched_task_record_count', 0)} |",
        f"| `completed_task_count` | {report.get('completed_task_count', 0)} |",
        f"| `progress_observation_count` | {report.get('progress_observation_count', 0)} |",
        f"| `usable_for_live_completion_claim` | {str(bool(report.get('usable_for_live_completion_claim'))).lower()} |",
        f"| `usable_for_live_progress_claim` | {str(bool(report.get('usable_for_live_progress_claim'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    return "\n".join(lines)


def _slot_realization_row(slot: Mapping[str, Any], task: Mapping[str, Any] | None) -> dict[str, Any]:
    if task is None:
        return {
            "slot_id": slot.get("slot_id"),
            "task_id": slot.get("task_id"),
            "score_semantics": slot.get("score_semantics"),
            "task_record_found": False,
            "terminal_status": "",
            "progress_rate_per_s": 0.0,
            "flow_s": None,
        }
    obs = task_progress_observation(task)
    started = _as_float(task.get("started_at"), 0.0)
    ended = _as_float(task.get("ended_at", task.get("finished_at")), 0.0)
    submitted = _as_float(task.get("submitted_at"), 0.0)
    terminal = str(task.get("status") or "")
    flow = max(0.0, ended - submitted) if ended > 0.0 and submitted > 0.0 else None
    return {
        "slot_id": slot.get("slot_id"),
        "task_id": slot.get("task_id"),
        "score_semantics": slot.get("score_semantics"),
        "task_record_found": True,
        "terminal_status": terminal,
        "node": task.get("node"),
        "gpu_idx": task.get("gpu_idx"),
        "started_at": started or None,
        "ended_at": ended or None,
        "flow_s": flow,
        "progress_current": obs.current if obs else None,
        "progress_total": obs.total if obs else None,
        "progress_rate_per_s": float(obs.rate_per_s or 0.0) if obs else 0.0,
        "progress_unit": obs.unit if obs else "",
    }


def _load_records(*, queue_path: str | Path, archive_path: str | Path) -> list[dict[str, Any]]:
    rows = []
    q = Path(queue_path).expanduser()
    if q.exists():
        try:
            payload = json.loads(q.read_text(encoding="utf-8"))
            tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
            if isinstance(tasks, list):
                rows.extend(dict(row) for row in tasks if isinstance(row, Mapping))
        except Exception:
            pass
    a = Path(archive_path).expanduser()
    if a.exists():
        try:
            for line in a.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, Mapping):
                        rows.append(dict(row))
        except Exception:
            pass
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_realization_bridge(
        trace_path=args.input,
        queue_path=args.queue_path,
        archive_path=args.archive_path,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("usable_for_live_progress_claim") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.live_trace_realization_bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Match theorem trace slots to realized scheduler records")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--markdown-output", default="")
    build.add_argument("--queue-path", default=str(DEFAULT_QUEUE))
    build.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
