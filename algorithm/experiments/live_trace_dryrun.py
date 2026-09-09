"""Generate live-state scheduler candidate traces without dispatching tasks.

This probe imports the existing legacy scheduler module, uses its current node
probe and ``pick_placement`` implementation, and writes oracle trace slots via
the already-installed ``algorithm.oracle_trace`` hook.  It never mutates the
queue and never calls ``launch`` or ``_do_dispatch``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from algorithm.oracle_trace import load_trace_slots


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "skill" / "scheduler.py"
DEFAULT_TRACE_PATH = Path.home() / ".claude" / "scheduler" / "oracle_trace.jsonl"


def generate_dryrun_trace(
    *,
    trace_path: str | Path = DEFAULT_TRACE_PATH,
    report_path: str | Path | None = None,
    task_count: int = 24,
    algorithm: str = "sweetspot_v1",
    hard_rule_mode: str = "",
    append: bool = False,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    trace.parent.mkdir(parents=True, exist_ok=True)
    if not append and trace.exists():
        trace.unlink()
    old_trace_env = os.environ.get("SCHEDULEURM_ORACLE_TRACE_PATH")
    os.environ["SCHEDULEURM_ORACLE_TRACE_PATH"] = str(trace)
    scheduler = _load_scheduler_module()
    try:
        _configure_scheduler(scheduler, algorithm=algorithm, hard_rule_mode=hard_rule_mode)
        state = scheduler.load_state()
        nodes = scheduler.probe_all()
        _prepare_probe_nodes(scheduler, state, nodes)
        tasks = _synthetic_tasks(task_count)
        rows = []
        for task in tasks:
            placement = scheduler.pick_placement(task, nodes)
            rows.append({
                "task_id": task["id"],
                "signature": task["signature"],
                "resource_kind": "gpu" if int(task.get("est_vram_mb") or 0) > 0 else "cpu",
                "placement": {
                    "node": placement[0],
                    "gpu_idx": placement[1],
                } if placement else None,
            })
            if placement:
                _reserve_synthetic_capacity(scheduler, nodes, task, placement)
        slots = load_trace_slots(trace)
        report = {
            "gate": "live_trace_dryrun_probe",
            "trace_path": str(trace),
            "scheduler_path": str(SCHEDULER_PATH),
            "algorithm": _algorithm_name(scheduler),
            "requested_algorithm": algorithm,
            "hard_rule_mode": hard_rule_mode,
            "task_count": len(tasks),
            "placed_count": sum(1 for row in rows if row.get("placement")),
            "unplaced_count": sum(1 for row in rows if not row.get("placement")),
            "trace_slot_count": len(slots),
            "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
            "node_summary": _node_summary(nodes),
            "placements": rows,
            "pass": len(slots) > 0,
            "scope": (
                "synthetic task records against current live node probe; queue is "
                "not modified and no task is launched"
            ),
        }
        if report_path:
            _write_json(report_path, report)
        return report
    finally:
        if old_trace_env is None:
            os.environ.pop("SCHEDULEURM_ORACLE_TRACE_PATH", None)
        else:
            os.environ["SCHEDULEURM_ORACLE_TRACE_PATH"] = old_trace_env


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live Trace Dry-Run Probe",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| trace_slot_count | {report.get('trace_slot_count', 0)} |",
        f"| candidate_count_total | {report.get('candidate_count_total', 0)} |",
        f"| placed_count | {report.get('placed_count', 0)} |",
        f"| unplaced_count | {report.get('unplaced_count', 0)} |",
        f"| algorithm | `{report.get('algorithm', '')}` |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Placements",
        "",
        "| task | kind | placement |",
        "|---|---|---|",
    ]
    for row in report.get("placements") or []:
        lines.append(
            f"| `{row.get('task_id')}` | {row.get('resource_kind')} | "
            f"`{json.dumps(row.get('placement'), sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_scheduler_module():
    spec = importlib.util.spec_from_file_location("scheduleurm_legacy_scheduler_for_probe", SCHEDULER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scheduler from {SCHEDULER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_scheduler(module: Any, *, algorithm: str, hard_rule_mode: str) -> None:
    if hasattr(module, "_set_hard_rule_mode"):
        module._set_hard_rule_mode(hard_rule_mode or None)
    if hasattr(module, "_configure_algorithm"):
        module._configure_algorithm(algorithm or None)


def _prepare_probe_nodes(module: Any, state: Mapping[str, Any], nodes: list[dict[str, Any]]) -> None:
    if hasattr(module, "_apply_cpu_slot_accounting_to_nodes"):
        module._apply_cpu_slot_accounting_to_nodes(state, nodes)
    running_per_node = Counter(
        task.get("node") for task in state.get("tasks", [])
        if _counts_against_concurrency(module, task)
    )
    running_per_gpu = Counter(
        (task.get("node"), task.get("gpu_idx")) for task in state.get("tasks", [])
        if _counts_against_concurrency(module, task) and task.get("gpu_idx") is not None
    )
    for node in nodes:
        node["running_count"] = running_per_node.get(node.get("name"), 0)
        for gpu in node.get("gpus") or []:
            gpu["running_task_count"] = running_per_gpu.get((node.get("name"), gpu.get("idx")), 0)


def _counts_against_concurrency(module: Any, task: Mapping[str, Any]) -> bool:
    if hasattr(module, "_counts_against_node_concurrency"):
        try:
            return bool(module._counts_against_node_concurrency(task))
        except Exception:
            return task.get("status") in ("running", "launching")
    return task.get("status") in ("running", "launching")


def _synthetic_tasks(task_count: int) -> list[dict[str, Any]]:
    now = time.time()
    tasks = []
    gpu_budget = min(6, max(0, task_count // 4))
    for idx in range(max(0, int(task_count))):
        gpu_task = idx < gpu_budget
        tasks.append({
            "id": f"dryrun-oracle-{idx:03d}",
            "status": "queued",
            "submitted_at": now + idx * 0.001,
            "priority": ("high" if idx % 11 == 0 else "normal"),
            "description": (
                "synthetic live oracle GPU placement probe"
                if gpu_task else
                "synthetic live oracle CPU placement probe"
            ),
            "cmd": "true",
            "cwd": str(REPO_ROOT),
            "signature": f"or-live-trace/{'gpu' if gpu_task else 'cpu'}/{idx:03d}",
            "project": "or-live-trace",
            "est_vram_mb": 512 if gpu_task else 0,
            "ram_mb": 2048 if gpu_task else 512,
            "cpu_cores": 1,
            "allowed_nodes": [],
            "preferred_node": None,
            "require_node": None,
            "env": [],
        })
    return tasks


def _reserve_synthetic_capacity(
    module: Any,
    nodes: list[dict[str, Any]],
    task: Mapping[str, Any],
    placement: tuple[Any, Any],
) -> None:
    node_name, gpu_idx = placement
    for node in nodes:
        if node.get("name") != node_name:
            continue
        cpu = int(task.get("cpu_cores") or getattr(module, "DEFAULT_CPU_CORES", 1))
        ram = int(task.get("ram_mb") or getattr(module, "DEFAULT_RAM_MB", 4096))
        node["free_cpu"] = max(0, int(node.get("free_cpu") or 0) - cpu)
        node["free_ram_mb"] = max(0, int(node.get("free_ram_mb") or 0) - ram)
        node["running_count"] = int(node.get("running_count") or 0) + 1
        if gpu_idx is None:
            return
        for gpu in node.get("gpus") or []:
            try:
                same_gpu = int(gpu.get("idx")) == int(gpu_idx)
            except Exception:
                same_gpu = str(gpu.get("idx")) == str(gpu_idx)
            if not same_gpu:
                continue
            need = int(task.get("est_vram_mb") or 0)
            gpu["used_mb"] = int(gpu.get("used_mb") or 0) + need
            gpu["free_mb"] = max(0, int(gpu.get("free_mb") or 0) - need)
            gpu["running_task_count"] = int(gpu.get("running_task_count") or 0) + 1
            return


def _node_summary(nodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for node in nodes:
        out.append({
            "name": node.get("name"),
            "alive": bool(node.get("alive")),
            "free_cpu": node.get("free_cpu"),
            "free_ram_mb": node.get("free_ram_mb"),
            "gpu_count": len(node.get("gpus") or []),
        })
    return out


def _algorithm_name(module: Any) -> str:
    if hasattr(module, "_algorithm_name"):
        try:
            return str(module._algorithm_name())
        except Exception:
            return ""
    return ""


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_probe(args: argparse.Namespace) -> int:
    report = generate_dryrun_trace(
        trace_path=args.trace_path,
        report_path=args.output,
        task_count=args.task_count,
        algorithm=args.algorithm,
        hard_rule_mode=args.hard_rule_mode,
        append=args.append,
    )
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.live_trace_dryrun")
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="Generate live-state dry-run placement trace")
    probe.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    probe.add_argument("--output", default=str(REPO_ROOT / "md" / "experiment_artifacts" / "or_gate_live_trace_dryrun.json"))
    probe.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_live_trace_dryrun.md"))
    probe.add_argument("--task-count", type=int, default=24)
    probe.add_argument("--algorithm", default="sweetspot_v1")
    probe.add_argument("--hard-rule-mode", default="")
    probe.add_argument("--append", action="store_true")
    probe.set_defaults(func=_cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
