"""Generate a theorem trace for currently queued production tasks.

This is a read-only live-state probe.  It uses real queued production records as
the task population, current node probes as the resource population, and the
optional ``theorem_maxweight_v1`` hook as the placement oracle.  It does not
mutate ``queue.json`` and does not launch jobs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from algorithm.oracle_trace import load_trace_slots
from algorithm.theorem_dispatch.admission import task_admission_certificate

from .live_trace_dryrun import (
    REPO_ROOT,
    _algorithm_name,
    _configure_scheduler,
    _load_scheduler_module,
    _node_summary,
    _prepare_probe_nodes,
    _reserve_synthetic_capacity,
    _write_json,
)
from .production_load_certificate import load_scheduler_records, _dedupe_records
from .production_live_theorem_trace_gate import _looks_like_production
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRACE = ARTIFACT_ROOT / "production_queued_theorem_trace.jsonl"


def generate_production_queued_theorem_trace(
    *,
    trace_path: str | Path = DEFAULT_TRACE,
    output_path: str | Path | None = None,
    max_tasks: int = 64,
    max_tasks_per_gpu: int = 8,
    algorithm: str = "theorem_maxweight_v1",
    append: bool = False,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    trace.parent.mkdir(parents=True, exist_ok=True)
    if not append and trace.exists():
        trace.unlink()

    queued_rows = _queued_admissible_production(load_scheduler_records(), max_tasks=max_tasks)
    old_env = {
        "SCHEDULEURM_ORACLE_TRACE_PATH": os.environ.get("SCHEDULEURM_ORACLE_TRACE_PATH"),
        "SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU": os.environ.get("SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"),
        "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": os.environ.get("SCHEDULEURM_THEOREM_UNCERTIFIED_MODE"),
    }
    os.environ["SCHEDULEURM_ORACLE_TRACE_PATH"] = str(trace)
    os.environ["SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"] = str(max(1, int(max_tasks_per_gpu)))
    os.environ["SCHEDULEURM_THEOREM_UNCERTIFIED_MODE"] = "block"

    scheduler = _load_scheduler_module()
    try:
        _configure_scheduler(scheduler, algorithm=algorithm, hard_rule_mode="production-queued")
        state = scheduler.load_state()
        nodes = scheduler.probe_all()
        _prepare_probe_nodes(scheduler, state, nodes)
        placements = []
        for row in queued_rows:
            task = _trace_task(row)
            placement = scheduler.pick_placement(task, nodes)
            placements.append({
                "task_id": task["id"],
                "project": task.get("project"),
                "signature": task.get("signature"),
                "workload_key": task.get("workload_key"),
                "placement": {
                    "node": placement[0],
                    "gpu_idx": placement[1],
                } if placement else None,
            })
            if placement:
                _reserve_synthetic_capacity(scheduler, nodes, task, placement)
        slots = load_trace_slots(trace)
        bridge = build_theorem_oracle_audit_from_trace(slots)
        report = {
            "gate": "production_queued_theorem_trace",
            "trace_path": str(trace),
            "algorithm": _algorithm_name(scheduler),
            "queued_admissible_count": len(queued_rows),
            "max_tasks": int(max_tasks),
            "max_tasks_per_gpu": int(max_tasks_per_gpu),
            "requested_algorithm": algorithm,
            "placed_count": sum(1 for row in placements if row.get("placement")),
            "unplaced_count": sum(1 for row in placements if not row.get("placement")),
            "trace_slot_count": len(slots),
            "theorem_slot_count": sum(
                1 for slot in slots
                if slot.get("score_semantics") == "robust_maxweight_lower_service"
            ),
            "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
            "oracle_bridge": bridge,
            "node_summary": _node_summary(nodes),
            "placements": placements,
            "pass": bool(queued_rows)
            and all(row.get("placement") for row in placements)
            and bool(bridge.get("usable_for_theorem"))
            and len(slots) == len(placements),
            "scope": (
                "read-only theorem trace over currently queued production tasks; "
                "queue is not mutated and no process is launched."
            ),
        }
        if output_path:
            _write_json(output_path, report)
        return report
    finally:
        _restore_env(old_env)


def markdown_report(report: Mapping[str, Any]) -> str:
    bridge = report.get("oracle_bridge") or {}
    lines = [
        "# Production Queued Theorem Trace",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `algorithm` | `{report.get('algorithm', '')}` |",
        f"| `queued_admissible_count` | {report.get('queued_admissible_count', 0)} |",
        f"| `placed_count` | {report.get('placed_count', 0)} |",
        f"| `unplaced_count` | {report.get('unplaced_count', 0)} |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `theorem_slot_count` | {report.get('theorem_slot_count', 0)} |",
        f"| `candidate_count_total` | {report.get('candidate_count_total', 0)} |",
        f"| `bridge_usable_for_theorem` | {str(bool(bridge.get('usable_for_theorem'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Placements",
        "",
        "| Task | Project | Workload | Placement |",
        "|---|---|---|---|",
    ]
    for row in report.get("placements") or []:
        lines.append(
            f"| `{row.get('task_id')}` | `{row.get('project')}` | "
            f"`{row.get('workload_key')}` | `{json.dumps(row.get('placement'), sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _queued_admissible_production(records: list[Mapping[str, Any]], *, max_tasks: int) -> list[dict[str, Any]]:
    rows = []
    for row in _dedupe_records(records):
        if str(row.get("status") or "") != "queued":
            continue
        if not _looks_like_production(row):
            continue
        cert = task_admission_certificate(row)
        if not cert.admitted:
            continue
        out = dict(row)
        out["workload_key"] = cert.workload_key
        rows.append(out)
    rows.sort(key=lambda row: float(row.get("submitted_at") or row.get("time") or 0.0))
    return rows[: max(0, int(max_tasks))]


def _trace_task(row: Mapping[str, Any]) -> dict[str, Any]:
    workload = str(row.get("workload_key") or "hybrid_rl_resac_ant")
    task = dict(row)
    task["id"] = str(row.get("id") or row.get("task_id") or "")
    task["status"] = "queued"
    task["workload_key"] = workload
    task.setdefault("description", "production queued theorem trace probe")
    task.setdefault("cwd", str(REPO_ROOT))
    task.setdefault("cmd", row.get("cmd") or row.get("command") or _default_cmd(workload))
    task.setdefault("signature", row.get("signature") or f"{workload}/production_queued")
    task.setdefault("project", row.get("project") or "production")
    task.setdefault("est_vram_mb", 512 if workload == "hybrid_rl_resac_ant" else 1024)
    task.setdefault("ram_mb", 2048)
    task.setdefault("cpu_cores", 1)
    task.setdefault("allowed_nodes", [])
    task.setdefault("preferred_node", None)
    task.setdefault("require_node", None)
    task.setdefault("env", [])
    return task


def _default_cmd(workload: str) -> str:
    if workload == "gpu_heavy_jax_matmul":
        return "python3 jax_matmul_size8192.py --size 8192 --steps 2400"
    if workload == "cpu_heavy_local_bench":
        return "python3 cpu_heavy_local_bench.py --units 1000"
    return "python3 train_resac_ant.py --env Ant-v4 --progress-units 80"


def _restore_env(old_env: Mapping[str, str | None]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _cmd_probe(args: argparse.Namespace) -> int:
    report = generate_production_queued_theorem_trace(
        trace_path=args.trace_path,
        output_path=args.output,
        max_tasks=args.max_tasks,
        max_tasks_per_gpu=args.max_tasks_per_gpu,
        algorithm=args.algorithm,
        append=args.append,
    )
    if args.markdown_output:
        path = Path(args.markdown_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.production_queued_theorem_trace")
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="Generate theorem trace from current queued production tasks")
    probe.add_argument("--trace-path", default=str(DEFAULT_TRACE))
    probe.add_argument("--output", default=str(ARTIFACT_ROOT / "production_queued_theorem_trace.json"))
    probe.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "production_queued_theorem_trace.md"))
    probe.add_argument("--max-tasks", type=int, default=64)
    probe.add_argument("--max-tasks-per-gpu", type=int, default=8)
    probe.add_argument("--algorithm", default="theorem_maxweight_v1")
    probe.add_argument("--append", action="store_true")
    probe.set_defaults(func=_cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
