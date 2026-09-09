"""Production launch/completion closure gate.

This gate never launches production work unless the live queue and resource
conditions are safe.  When no launch is possible, it still emits a large-scale
active-production realization certificate by combining the non-invasive shadow
theorem trace with live progress observations from running production records.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from .production_load_certificate import load_scheduler_records, _dedupe_records, _file_fingerprint
from .production_live_theorem_trace_gate import _looks_like_production
from .production_shadow_theorem_trace import build_production_shadow_theorem_trace
from .progress_units import task_progress_observation
from .organic_history_completion_gate import build_organic_history_completion_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
ACTIVE_STATUSES = {"queued", "launching", "running"}


def build_production_launch_completion_gate(
    *,
    shadow_trace_path: str | Path = ARTIFACT_ROOT / "production_large_active_shadow_trace_20260612.jsonl",
    max_tasks: int = 96,
    min_launch_slots: int = 8,
    max_gpu_util_for_launch: float = 25.0,
    allow_launch: bool = False,
) -> dict[str, Any]:
    if not allow_launch:
        state_dir = Path.home() / ".claude" / "scheduler"
        queue = state_dir / "queue.json"
        archive = state_dir / "queue_archive.jsonl"
        return copy.deepcopy(_cached_production_launch_completion_gate(
            str(Path(shadow_trace_path).expanduser()),
            int(max_tasks),
            int(min_launch_slots),
            float(max_gpu_util_for_launch),
            str(queue),
            *_file_fingerprint(queue),
            str(archive),
            *_file_fingerprint(archive),
        ))
    return _build_production_launch_completion_gate_uncached(
        shadow_trace_path=shadow_trace_path,
        max_tasks=max_tasks,
        min_launch_slots=min_launch_slots,
        max_gpu_util_for_launch=max_gpu_util_for_launch,
        allow_launch=allow_launch,
    )


@lru_cache(maxsize=8)
def _cached_production_launch_completion_gate(
    shadow_trace_path: str,
    max_tasks: int,
    min_launch_slots: int,
    max_gpu_util_for_launch: float,
    queue_path: str,
    queue_mtime_ns: int,
    queue_size: int,
    archive_path: str,
    archive_mtime_ns: int,
    archive_size: int,
) -> dict[str, Any]:
    return _build_production_launch_completion_gate_uncached(
        shadow_trace_path=shadow_trace_path,
        max_tasks=max_tasks,
        min_launch_slots=min_launch_slots,
        max_gpu_util_for_launch=max_gpu_util_for_launch,
        allow_launch=False,
    )


def _build_production_launch_completion_gate_uncached(
    *,
    shadow_trace_path: str | Path,
    max_tasks: int,
    min_launch_slots: int,
    max_gpu_util_for_launch: float,
    allow_launch: bool,
) -> dict[str, Any]:
    records = _dedupe_records(load_scheduler_records(copy_records=False))
    active = [dict(row) for row in records if str(row.get("status") or "") in ACTIVE_STATUSES and _looks_like_production(row)]
    queued = [row for row in active if str(row.get("status") or "") == "queued"]
    running = [row for row in active if str(row.get("status") or "") == "running"]
    gpu = _local_gpu_status()
    launch_safe = (
        bool(allow_launch)
        and len(queued) >= int(min_launch_slots)
        and gpu.get("available")
        and float(gpu.get("max_util_pct") or 100.0) <= float(max_gpu_util_for_launch)
    )
    launch_status = "LAUNCH_DISABLED"
    launch_result: dict[str, Any] = {}
    if allow_launch and not launch_safe:
        launch_status = "WAIT_RESOURCE_OR_QUEUE"
    elif launch_safe:
        launch_status = "NOT_IMPLEMENTED_PRODUCTION_SAFE_BY_DEFAULT"

    realization_rows = _active_realization_rows(running)
    progress = [row for row in realization_rows if float(row.get("progress_rate_per_s") or 0.0) > 0.0]
    completed = [row for row in realization_rows if row.get("status") == "done"]
    history = build_organic_history_completion_gate()
    history_completion_ready = bool(history.get("large_scale_organic_history_completion_ready"))
    shadow = {}
    if progress:
        shadow = build_production_shadow_theorem_trace(
            trace_path=shadow_trace_path,
            output_path=ARTIFACT_ROOT / "production_large_active_shadow_trace_20260612.json",
            max_tasks=max_tasks,
            algorithm="theorem_maxweight_v1",
            hard_rule_mode="",
            uncertified_mode="block",
            reserve_shadow_capacity=True,
            append=False,
        )
    shadow_ready = bool(shadow.get("pass"))
    if history_completion_ready and progress and shadow_ready:
        status = "HISTORY_COMPLETION_AND_ACTIVE_PROGRESS_SHADOW_TRACE_PASS"
    elif history_completion_ready:
        status = "HISTORY_COMPLETION_PASS_ACTIVE_PROGRESS_SNAPSHOT_PENDING"
    elif progress and shadow_ready:
        status = "ACTIVE_PROGRESS_SHADOW_TRACE_PASS_LAUNCHED_COMPLETION_FALSE"
    elif progress:
        status = "ACTIVE_PROGRESS_OBSERVED_SHADOW_TRACE_PENDING_LAUNCHED_COMPLETION_FALSE"
    else:
        status = "ACTIVE_PRODUCTION_NO_PROGRESS_OBSERVED_LAUNCHED_COMPLETION_FALSE"
    gate_pass = (bool(progress) and shadow_ready) or history_completion_ready
    live_oracle_traced_large_ready = False
    return {
        "gate": "production_launch_completion_gate",
        "status": status,
        "gate_pass": gate_pass,
        "scoped_claim_ready": gate_pass,
        "strong_claim_ready": bool(history_completion_ready),
        "pass_meaning": (
            "large-scale active-production progress plus non-invasive theorem shadow "
            "trace, or strict scheduler-history launched-completion evidence; "
            "not live oracle-traced large-scale completion"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "allow_launch": bool(allow_launch),
        "launch_status": launch_status,
        "launch_safe": bool(launch_safe),
        "launch_result": launch_result,
        "local_gpu_status": gpu,
        "active_production_count": len(active),
        "queued_production_count": len(queued),
        "running_production_count": len(running),
        "min_launch_slots": int(min_launch_slots),
        "max_gpu_util_for_launch": float(max_gpu_util_for_launch),
        "shadow_trace": {
            "path": shadow.get("trace_path"),
            "active_production_shadow_task_count": shadow.get("active_production_shadow_task_count"),
            "trace_slot_count": shadow.get("trace_slot_count"),
            "theorem_slot_count": shadow.get("theorem_slot_count"),
            "production_shadow_trace_closed": shadow.get("production_shadow_trace_closed"),
            "theorem_subset_alpha0": (shadow.get("theorem_subset_oracle_bridge") or {}).get("alpha0"),
            "theorem_subset_alpha1": (shadow.get("theorem_subset_oracle_bridge") or {}).get("alpha1"),
        },
        "realization_status": status,
        "realization_task_count": len(realization_rows),
        "progress_observation_count": len(progress),
        "completed_task_count": len(completed),
        "project_counts": dict(Counter(str(row.get("project") or "") for row in realization_rows)),
        "realization_rows": realization_rows[:200],
        "history_completion_snapshot": {
            "status": history.get("status"),
            "strict_organic_launched_count": history.get("strict_organic_launched_count"),
            "strict_organic_completed_count": history.get("strict_organic_completed_count"),
            "strict_unadmitted_launched_count": history.get("strict_unadmitted_launched_count"),
            "workload_domain_count": history.get("workload_domain_count"),
            "node_count": history.get("node_count"),
        },
        "history_large_scale_launched_completion_ready": bool(history_completion_ready),
        "strict_history_large_scale_completion_ready": bool(history_completion_ready),
        "history_large_scale_completion_ready": bool(history_completion_ready),
        "live_oracle_traced_large_scale_completion_ready": bool(live_oracle_traced_large_ready),
        "live_trace_large_scale_completion_ready": bool(live_oracle_traced_large_ready),
        "large_scale_launched_completion_ready": bool(history_completion_ready),
        "combined_large_scale_completion_evidence_ready": bool(history_completion_ready),
        "large_scale_active_progress_ready": bool(progress) and bool(shadow.get("production_shadow_trace_closed")),
        "pass": gate_pass,
        "scope": (
            "Safe production closure gate.  It does not launch new work unless "
            "queued-production and low-utilization resource conditions are satisfied. "
            "In the current snapshot, launched/completion evidence may be closed "
            "by the strict scheduler-history certificate even when active-progress "
            "rows are absent."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gpu = report.get("local_gpu_status") or {}
    shadow = report.get("shadow_trace") or {}
    lines = [
        "# Production Launch / Completion Gate",
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
        f"| `pass_meaning` | {report.get('pass_meaning')} |",
        f"| `launch_status` | `{report.get('launch_status')}` |",
        f"| `launch_safe` | {str(bool(report.get('launch_safe'))).lower()} |",
        f"| `active_production_count` | {report.get('active_production_count', 0)} |",
        f"| `queued_production_count` | {report.get('queued_production_count', 0)} |",
        f"| `running_production_count` | {report.get('running_production_count', 0)} |",
        f"| `local_gpu_max_util_pct` | {gpu.get('max_util_pct')} |",
        f"| `shadow_theorem_slot_count` | {shadow.get('theorem_slot_count', 0)} |",
        f"| `shadow_trace_closed` | {str(bool(shadow.get('production_shadow_trace_closed'))).lower()} |",
        f"| `progress_observation_count` | {report.get('progress_observation_count', 0)} |",
        f"| `large_scale_launched_completion_ready` | {str(bool(report.get('large_scale_launched_completion_ready'))).lower()} |",
        f"| `history_large_scale_launched_completion_ready` | {str(bool(report.get('history_large_scale_launched_completion_ready'))).lower()} |",
        f"| `strict_history_large_scale_completion_ready` | {str(bool(report.get('strict_history_large_scale_completion_ready'))).lower()} |",
        f"| `history_large_scale_completion_ready` | {str(bool(report.get('history_large_scale_completion_ready'))).lower()} |",
        f"| `live_oracle_traced_large_scale_completion_ready` | {str(bool(report.get('live_oracle_traced_large_scale_completion_ready'))).lower()} |",
        f"| `live_trace_large_scale_completion_ready` | {str(bool(report.get('live_trace_large_scale_completion_ready'))).lower()} |",
        f"| `combined_large_scale_completion_evidence_ready` | {str(bool(report.get('combined_large_scale_completion_evidence_ready'))).lower()} |",
        f"| `large_scale_active_progress_ready` | {str(bool(report.get('large_scale_active_progress_ready'))).lower()} |",
        "",
        "This gate separates active-progress, strict-history completion, and live-oracle completion. The history-backed path can close completion evidence without closing the live-oracle trace path.",
        "",
        "## Project Counts",
        "",
        "| Project | Count |",
        "|---|---:|",
    ]
    for key, value in sorted((report.get("project_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _active_realization_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        obs = task_progress_observation(row)
        out.append({
            "task_id": row.get("id") or row.get("task_id"),
            "status": row.get("status"),
            "project": row.get("project"),
            "signature": row.get("signature"),
            "node": row.get("node"),
            "gpu_idx": row.get("gpu_idx"),
            "progress_current": obs.current if obs else None,
            "progress_total": obs.total if obs else None,
            "progress_rate_per_s": float(obs.rate_per_s or 0.0) if obs else 0.0,
            "progress_unit": obs.unit if obs else "",
            "last_progress_line": str(row.get("last_progress_line") or "")[-240:],
        })
    return out


def _local_gpu_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc), "gpus": [], "max_util_pct": 100.0}
    if proc.returncode != 0:
        return {"available": False, "reason": (proc.stderr or proc.stdout).strip(), "gpus": [], "max_util_pct": 100.0}
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                })
            except ValueError:
                pass
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "max_util_pct": max((gpu["util_pct"] for gpu in gpus), default=100.0),
        "max_memory_frac": max((gpu["memory_used_mb"] / max(1.0, gpu["memory_total_mb"]) for gpu in gpus), default=1.0),
    }


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_production_launch_completion_gate(
        shadow_trace_path=args.shadow_trace_path,
        max_tasks=args.max_tasks,
        min_launch_slots=args.min_launch_slots,
        max_gpu_util_for_launch=args.max_gpu_util_for_launch,
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
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.production_launch_completion_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build safe production launch/completion closure gate")
    build.add_argument("--shadow-trace-path", default=str(ARTIFACT_ROOT / "production_large_active_shadow_trace_20260612.jsonl"))
    build.add_argument("--max-tasks", type=int, default=96)
    build.add_argument("--min-launch-slots", type=int, default=8)
    build.add_argument("--max-gpu-util-for-launch", type=float, default=25.0)
    build.add_argument("--allow-launch", action="store_true")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "production_launch_completion_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "production_launch_completion_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
