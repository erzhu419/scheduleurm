"""Generate theorem-grade candidate traces from live node probes.

This runner exercises the optional ``theorem_maxweight_v1`` scheduler hook
without mutating ``queue.json`` or launching jobs.  It differs from the older
dry-run trace in two ways:

* the synthetic tasks are measured benchmark workloads from the service cache;
* uncertified candidates are blocked, so every emitted slot can be consumed
  directly by ``theorem_oracle_trace_bridge``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.oracle_trace import load_trace_slots

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
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRACE = ARTIFACT_ROOT / "natural_live_theorem_trace.jsonl"


def generate_natural_live_theorem_trace(
    *,
    trace_path: str | Path = DEFAULT_TRACE,
    output_path: str | Path | None = None,
    task_count: int = 16,
    workloads: Iterable[str] = ("hybrid_rl_resac_ant", "gpu_heavy_jax_matmul"),
    max_tasks_per_gpu: int = 8,
    algorithm: str = "theorem_maxweight_v1",
    append: bool = False,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    trace.parent.mkdir(parents=True, exist_ok=True)
    if not append and trace.exists():
        trace.unlink()

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
        _configure_scheduler(scheduler, algorithm=algorithm, hard_rule_mode="theorem-natural")
        state = scheduler.load_state()
        nodes = scheduler.probe_all()
        _prepare_probe_nodes(scheduler, state, nodes)
        tasks = _measured_gpu_tasks(task_count, tuple(workloads))
        placements = []
        for task in tasks:
            placement = scheduler.pick_placement(task, nodes)
            placements.append({
                "task_id": task["id"],
                "workload_key": task["workload_key"],
                "signature": task["signature"],
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
            "gate": "natural_live_theorem_trace",
            "trace_path": str(trace),
            "algorithm": _algorithm_name(scheduler),
            "task_count": len(tasks),
            "requested_workloads": list(workloads),
            "max_tasks_per_gpu": int(max_tasks_per_gpu),
            "requested_algorithm": algorithm,
            "placed_count": sum(1 for row in placements if row.get("placement")),
            "unplaced_count": sum(1 for row in placements if not row.get("placement")),
            "trace_slot_count": len(slots),
            "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
            "theorem_slot_count": sum(
                1 for slot in slots
                if slot.get("score_semantics") == "robust_maxweight_lower_service"
            ),
            "oracle_bridge": bridge,
            "node_summary": _node_summary(nodes),
            "placements": placements,
            "pass": bool(placements)
            and all(row.get("placement") for row in placements)
            and bool(bridge.get("usable_for_theorem"))
            and len(slots) == len(placements),
            "scope": (
                "live-node-probe dry run with measured q01/q11 workload records; "
                "queue is not mutated and no process is launched. Uncertified "
                "candidate profiles are blocked by the theorem policy during "
                "this probe."
            ),
        }
        if output_path:
            _write_json(output_path, report)
        return report
    finally:
        _restore_env(old_env)


def markdown_report(report: Mapping[str, Any]) -> str:
    bridge = report.get("oracle_bridge") or {}
    rows = [
        "# Natural Live Theorem Trace",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `algorithm` | `{report.get('algorithm', '')}` |",
        f"| `task_count` | {report.get('task_count', 0)} |",
        f"| `placed_count` | {report.get('placed_count', 0)} |",
        f"| `unplaced_count` | {report.get('unplaced_count', 0)} |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `theorem_slot_count` | {report.get('theorem_slot_count', 0)} |",
        f"| `candidate_count_total` | {report.get('candidate_count_total', 0)} |",
        f"| `alpha0` | {_fmt(bridge.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(bridge.get('alpha1'))} |",
        f"| `bridge_usable_for_theorem` | {str(bool(bridge.get('usable_for_theorem'))).lower()} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Placements",
        "",
        "| Task | Workload | Placement |",
        "|---|---|---|",
    ]
    for row in report.get("placements") or []:
        rows.append(
            f"| `{row.get('task_id')}` | `{row.get('workload_key')}` | "
            f"`{json.dumps(row.get('placement'), sort_keys=True)}` |"
        )
    rows.append("")
    return "\n".join(rows)


def _measured_gpu_tasks(task_count: int, workloads: tuple[str, ...]) -> list[dict[str, Any]]:
    now = time.time()
    keys = workloads or ("hybrid_rl_resac_ant",)
    tasks = []
    for idx in range(max(0, int(task_count))):
        key = keys[idx % len(keys)]
        tasks.append(_task_for_workload(key, idx, now + idx * 0.001))
    return tasks


def _task_for_workload(workload_key: str, idx: int, submitted_at: float) -> dict[str, Any]:
    if workload_key == "gpu_heavy_jax_matmul":
        project = "q01-gpu-heavy"
        signature = f"jax_matmul_size8192_v1/live_theorem/{idx:03d}"
        cmd = "python3 jax_matmul_size8192.py --size 8192 --steps 2400"
        desc = "q01 GPU-heavy JAX matmul theorem trace probe"
        total_vram = 1024
    else:
        project = "q11-hybrid-rl"
        signature = f"resac_ant_real_v1/live_theorem/{idx:03d}"
        cmd = "python3 train_resac_ant.py --env Ant-v4 --progress-units 80"
        desc = "q11 RE-SAC/BAPR-like hybrid RL theorem trace probe"
        total_vram = 512
    return {
        "id": f"natural-theorem-{idx:03d}",
        "status": "queued",
        "submitted_at": submitted_at,
        "priority": "normal",
        "description": desc,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "signature": signature,
        "project": project,
        "workload_key": workload_key,
        "est_vram_mb": total_vram,
        "ram_mb": 2048,
        "cpu_cores": 1,
        "allowed_nodes": [],
        "preferred_node": None,
        "require_node": None,
        "env": [],
    }


def _restore_env(old_env: Mapping[str, str | None]) -> None:
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return "NA"


def _cmd_probe(args: argparse.Namespace) -> int:
    report = generate_natural_live_theorem_trace(
        trace_path=args.trace_path,
        output_path=args.output,
        task_count=args.task_count,
        workloads=[x.strip() for x in args.workloads.split(",") if x.strip()],
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
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.natural_live_theorem_trace")
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="Generate theorem-grade live-node candidate trace")
    probe.add_argument("--trace-path", default=str(DEFAULT_TRACE))
    probe.add_argument("--output", default=str(ARTIFACT_ROOT / "natural_live_theorem_trace.json"))
    probe.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "natural_live_theorem_trace.md"))
    probe.add_argument("--task-count", type=int, default=16)
    probe.add_argument("--workloads", default="hybrid_rl_resac_ant,gpu_heavy_jax_matmul")
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
