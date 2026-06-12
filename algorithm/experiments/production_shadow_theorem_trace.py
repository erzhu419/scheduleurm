"""Non-invasive production-state theorem trace.

This probe is deliberately weaker than a production-wide launched dispatch
trace, but it is safe to run while the scheduler is in use.  It reads the real
queue and current node probes, copies active production tasks into shadow queued
records, and asks the optional theorem placement hook what it would do now.  It
never writes ``queue.json``, never calls dispatch, and never launches a process.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from .production_live_theorem_trace_gate import _looks_like_production
from .theorem_oracle_trace_bridge import build_theorem_oracle_audit_from_trace


ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TRACE = ARTIFACT_ROOT / "production_shadow_theorem_trace.jsonl"
ACTIVE_STATUSES = {"queued", "launching", "running"}


def build_production_shadow_theorem_trace(
    *,
    trace_path: str | Path = DEFAULT_TRACE,
    output_path: str | Path | None = None,
    max_tasks: int = 64,
    algorithm: str = "theorem_maxweight_v1",
    hard_rule_mode: str = "",
    uncertified_mode: str = "block",
    reserve_shadow_capacity: bool = True,
    append: bool = False,
) -> dict[str, Any]:
    trace = Path(trace_path).expanduser()
    trace.parent.mkdir(parents=True, exist_ok=True)
    if not append and trace.exists():
        trace.unlink()

    old_env = {
        "SCHEDULEURM_ORACLE_TRACE_PATH": os.environ.get("SCHEDULEURM_ORACLE_TRACE_PATH"),
        "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE": os.environ.get("SCHEDULEURM_THEOREM_UNCERTIFIED_MODE"),
        "SCHEDULEURM_AB_HARD_RULE_MODE": os.environ.get("SCHEDULEURM_AB_HARD_RULE_MODE"),
    }
    os.environ["SCHEDULEURM_ORACLE_TRACE_PATH"] = str(trace)
    os.environ["SCHEDULEURM_THEOREM_UNCERTIFIED_MODE"] = str(uncertified_mode or "block")

    scheduler = _load_scheduler_module()
    try:
        _configure_scheduler(scheduler, algorithm=algorithm, hard_rule_mode=hard_rule_mode)
        state = scheduler.load_state()
        nodes = scheduler.probe_all()
        _prepare_probe_nodes(scheduler, state, nodes)
        tasks = _shadow_tasks_from_state(state.get("tasks") or [], max_tasks=max_tasks)
        placements = []
        for task in tasks:
            placement = scheduler.pick_placement(task, nodes)
            placements.append({
                "shadow_task_id": task.get("id"),
                "source_task_id": task.get("source_task_id"),
                "source_status": task.get("source_status"),
                "project": task.get("project"),
                "signature": task.get("signature"),
                "inferred_workload_key": task.get("shadow_admission", {}).get("workload_key"),
                "admitted": task.get("shadow_admission", {}).get("admitted"),
                "placement": {
                    "node": placement[0],
                    "gpu_idx": placement[1],
                } if placement else None,
            })
            if placement and reserve_shadow_capacity:
                _reserve_synthetic_capacity(scheduler, nodes, task, placement)
        slots = load_trace_slots(trace)
        theorem_slots = [
            slot for slot in slots
            if slot.get("score_semantics") == "robust_maxweight_lower_service"
        ]
        bridge = build_theorem_oracle_audit_from_trace(slots)
        theorem_subset_bridge = build_theorem_oracle_audit_from_trace(theorem_slots)
        blocker_counts = Counter(
            _slot_blocker(slot) for slot in slots
            if slot.get("score_semantics") != "robust_maxweight_lower_service"
        )
        report = {
            "gate": "production_shadow_theorem_trace",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "trace_path": str(trace),
            "algorithm": _algorithm_name(scheduler),
            "requested_algorithm": algorithm,
            "hard_rule_mode": hard_rule_mode,
            "uncertified_mode": uncertified_mode,
            "reserve_shadow_capacity": bool(reserve_shadow_capacity),
            "active_production_shadow_task_count": len(tasks),
            "placed_count": sum(1 for row in placements if row.get("placement")),
            "unplaced_count": sum(1 for row in placements if not row.get("placement")),
            "trace_slot_count": len(slots),
            "theorem_slot_count": len(theorem_slots),
            "candidate_count_total": sum(len(slot.get("candidates") or []) for slot in slots),
            "oracle_bridge": bridge,
            "theorem_subset_oracle_bridge": theorem_subset_bridge,
            "non_theorem_blocker_counts": dict(blocker_counts.most_common()),
            "node_summary_after_shadow": _node_summary(nodes),
            "placements": placements,
            "production_shadow_trace_closed": bool(theorem_slots)
            and bool(theorem_subset_bridge.get("usable_for_theorem")),
            "launched_dispatch_claim": False,
            "pass": bool(tasks) and len(slots) > 0,
            "scope": (
                "non-invasive shadow trace over current active production tasks. "
                "It reads queue/node state and evaluates the optional theorem hook "
                "without queue mutation or launch.  It is not a production-wide "
                "launched dispatch trace."
            ),
        }
        if output_path:
            _write_json(output_path, report)
        return report
    finally:
        _restore_env(old_env)


def markdown_report(report: Mapping[str, Any]) -> str:
    bridge = report.get("oracle_bridge") or {}
    subset_bridge = report.get("theorem_subset_oracle_bridge") or {}
    rows = [
        "# Production Shadow Theorem Trace",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `production_shadow_trace_closed` | {str(bool(report.get('production_shadow_trace_closed'))).lower()} |",
        f"| `launched_dispatch_claim` | {str(bool(report.get('launched_dispatch_claim'))).lower()} |",
        f"| `active_production_shadow_task_count` | {report.get('active_production_shadow_task_count', 0)} |",
        f"| `placed_count` | {report.get('placed_count', 0)} |",
        f"| `unplaced_count` | {report.get('unplaced_count', 0)} |",
        f"| `trace_slot_count` | {report.get('trace_slot_count', 0)} |",
        f"| `theorem_slot_count` | {report.get('theorem_slot_count', 0)} |",
        f"| `candidate_count_total` | {report.get('candidate_count_total', 0)} |",
        f"| `bridge_usable_for_theorem` | {str(bool(bridge.get('usable_for_theorem'))).lower()} |",
        f"| `theorem_subset_bridge_usable` | {str(bool(subset_bridge.get('usable_for_theorem'))).lower()} |",
        f"| `alpha0` | {_fmt(bridge.get('alpha0'))} |",
        f"| `alpha1` | {_fmt(bridge.get('alpha1'))} |",
        f"| `theorem_subset_alpha0` | {_fmt(subset_bridge.get('alpha0'))} |",
        f"| `theorem_subset_alpha1` | {_fmt(subset_bridge.get('alpha1'))} |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
        "## Non-Theorem Blockers",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    blockers = report.get("non_theorem_blocker_counts") or {}
    if blockers:
        for key, value in blockers.items():
            rows.append(f"| `{key}` | {value} |")
    else:
        rows.append("| `none` | 0 |")
    rows.extend([
        "",
        "## Placements",
        "",
        "| Shadow task | Source task | Project | Workload | Admitted | Placement |",
        "|---|---|---|---|---:|---|",
    ])
    for row in report.get("placements") or []:
        rows.append(
            "| `{shadow}` | `{source}` | `{project}` | `{workload}` | {admitted} | `{placement}` |".format(
                shadow=row.get("shadow_task_id"),
                source=row.get("source_task_id"),
                project=row.get("project"),
                workload=row.get("inferred_workload_key"),
                admitted=str(bool(row.get("admitted"))).lower(),
                placement=json.dumps(row.get("placement"), sort_keys=True),
            )
        )
    rows.append("")
    return "\n".join(rows)


def _shadow_tasks_from_state(rows: Iterable[Mapping[str, Any]], *, max_tasks: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if len(out) >= max(0, int(max_tasks)):
            break
        if str(row.get("status") or "") not in ACTIVE_STATUSES:
            continue
        if not _looks_like_production(row):
            continue
        cert = task_admission_certificate(row)
        task = copy.deepcopy(dict(row))
        source_id = str(task.get("id") or task.get("task_id") or len(out))
        task["source_task_id"] = source_id
        task["source_status"] = task.get("status")
        task["id"] = f"shadow-production-{len(out):04d}-{source_id}"
        task["status"] = "queued"
        task["submitted_at"] = time.time() + len(out) * 0.001
        task.pop("node", None)
        task.pop("gpu_idx", None)
        task.pop("started_at", None)
        task.pop("pid", None)
        task["shadow_admission"] = cert.snapshot()
        out.append(task)
    return out


def _slot_blocker(slot: Mapping[str, Any]) -> str:
    candidates = list(slot.get("candidates") or [])
    if not candidates:
        return "no_candidates"
    reasons = Counter()
    for cand in candidates:
        audit = cand.get("algorithm_audit") or {}
        if isinstance(audit, Mapping):
            reason = str(audit.get("fallback_reason") or "")
            if reason:
                reasons[reason] += 1
    if reasons:
        return reasons.most_common(1)[0][0]
    return str(slot.get("score_semantics") or "non_theorem_semantics")


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
    report = build_production_shadow_theorem_trace(
        trace_path=args.trace_path,
        output_path=args.output,
        max_tasks=args.max_tasks,
        algorithm=args.algorithm,
        hard_rule_mode=args.hard_rule_mode,
        uncertified_mode=args.uncertified_mode,
        reserve_shadow_capacity=not args.no_reserve_shadow_capacity,
        append=args.append,
    )
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.production_shadow_theorem_trace"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="Generate non-invasive active-production shadow trace")
    probe.add_argument("--trace-path", default=str(DEFAULT_TRACE))
    probe.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "production_shadow_theorem_trace.json"),
    )
    probe.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "production_shadow_theorem_trace.md"),
    )
    probe.add_argument("--max-tasks", type=int, default=64)
    probe.add_argument("--algorithm", default="theorem_maxweight_v1")
    probe.add_argument("--hard-rule-mode", default="")
    probe.add_argument("--uncertified-mode", choices=("legacy", "block", "trace_only"), default="block")
    probe.add_argument("--no-reserve-shadow-capacity", action="store_true")
    probe.add_argument("--append", action="store_true")
    probe.set_defaults(func=_cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
