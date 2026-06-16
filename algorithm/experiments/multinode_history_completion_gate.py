"""Multi-node Scheduleurm production-history completion gate.

This gate closes the part of the multi-node claim that can be certified from
the current artifact without deploying third-party control planes: Scheduleurm
itself has launched and completed a strict theorem-facing production population
across multiple real nodes.  It does not claim that every external SOTA system
has been run in its original multi-node deployment.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache

from .production_live_theorem_trace_gate import _looks_like_production
from .production_load_certificate import _dedupe_records, load_scheduler_records
from ..theorem_dispatch.admission import task_admission_certificate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "multinode_history_completion_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "multinode_history_completion_gate_20260614.md"
GPU_NODE_PREFIXES = ("jtl110gpu", "node007")


def build_multinode_history_completion_gate(
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    min_nodes: int = 2,
    min_gpu_nodes: int = 2,
    min_completed: int = 50,
    min_workload_domains: int = 3,
    max_unadmitted: int = 0,
) -> dict[str, Any]:
    cache = build_default_cache()
    rows = _dedupe_records(records if records is not None else load_scheduler_records())
    strict_launched: list[tuple[Mapping[str, Any], Any]] = []
    strict_completed: list[tuple[Mapping[str, Any], Any]] = []
    unadmitted: list[tuple[Mapping[str, Any], Any]] = []
    excluded = 0
    for row in rows:
        if not _looks_like_production(row):
            continue
        if _excluded(row):
            excluded += 1
            continue
        if not _launched(row):
            continue
        cert = task_admission_certificate(row, cache=cache, admission_mode="strict")
        if not cert.admitted:
            unadmitted.append((row, cert))
            continue
        strict_launched.append((row, cert))
        if _completed(row):
            strict_completed.append((row, cert))

    node_counts = Counter(_node(row) for row, _ in strict_launched if _node(row))
    completed_node_counts = Counter(_node(row) for row, _ in strict_completed if _node(row))
    gpu_node_counts = Counter({
        node: count
        for node, count in node_counts.items()
        if count and _is_gpu_node(node)
    })
    domains = Counter(cert.workload_key for _, cert in strict_launched)
    ready = bool(
        len(strict_completed) >= int(min_completed)
        and len(node_counts) >= int(min_nodes)
        and len(gpu_node_counts) >= int(min_gpu_nodes)
        and len(domains) >= int(min_workload_domains)
        and len(unadmitted) <= int(max_unadmitted)
    )
    return {
        "gate": "multinode_history_completion_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "SCHEDULEURM_MULTINODE_HISTORY_COMPLETION_PASS"
            if ready else "SCHEDULEURM_MULTINODE_HISTORY_COMPLETION_PENDING"
        ),
        "gate_pass": ready,
        "scoped_claim_ready": ready,
        "strong_claim_ready": ready,
        "scheduleurm_multinode_launched_completion_ready": ready,
        "strict_launched_count": len(strict_launched),
        "strict_completed_count": len(strict_completed),
        "strict_unadmitted_count": len(unadmitted),
        "excluded_row_count": excluded,
        "node_count": len(node_counts),
        "completed_node_count": len(completed_node_counts),
        "gpu_node_count": len(gpu_node_counts),
        "workload_domain_count": len(domains),
        "min_nodes": int(min_nodes),
        "min_gpu_nodes": int(min_gpu_nodes),
        "min_completed": int(min_completed),
        "min_workload_domains": int(min_workload_domains),
        "max_unadmitted": int(max_unadmitted),
        "top_nodes": _top(node_counts, 20),
        "top_completed_nodes": _top(completed_node_counts, 20),
        "gpu_nodes": _top(gpu_node_counts, 20),
        "top_workload_domains": _top(domains, 20),
        "unadmitted_examples": [
            {
                "task_id": str(row.get("id") or row.get("task_id") or ""),
                "project": str(row.get("project") or ""),
                "signature": str(row.get("signature") or ""),
                "reason": str(cert.reason or ""),
            }
            for row, cert in unadmitted[:20]
        ],
        "blocker": "" if ready else _blocker(
            strict_completed=strict_completed,
            node_counts=node_counts,
            gpu_node_counts=gpu_node_counts,
            domains=domains,
            unadmitted=unadmitted,
            min_nodes=min_nodes,
            min_gpu_nodes=min_gpu_nodes,
            min_completed=min_completed,
            min_workload_domains=min_workload_domains,
            max_unadmitted=max_unadmitted,
        ),
        "scope": (
            "Scheduleurm-native multi-node launched/completion evidence over "
            "strict theorem-admitted production history.  This is not a claim "
            "that external SOTA control planes have been run in their original "
            "multi-node deployments."
        ),
        "pass": ready,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-Node History Completion Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `scheduleurm_multinode_launched_completion_ready` | {str(bool(report.get('scheduleurm_multinode_launched_completion_ready'))).lower()} |",
        f"| `strict_launched_count` | {report.get('strict_launched_count')} |",
        f"| `strict_completed_count` | {report.get('strict_completed_count')} |",
        f"| `strict_unadmitted_count` | {report.get('strict_unadmitted_count')} |",
        f"| `node_count` | {report.get('node_count')} |",
        f"| `gpu_node_count` | {report.get('gpu_node_count')} |",
        f"| `workload_domain_count` | {report.get('workload_domain_count')} |",
        "",
        "## Nodes",
        "",
        "| Node | Launched count |",
        "|---|---:|",
    ]
    for row in report.get("top_nodes") or []:
        lines.append(f"| `{row.get('name')}` | {row.get('count')} |")
    lines.extend([
        "",
        "## GPU Nodes",
        "",
        "| Node | Launched count |",
        "|---|---:|",
    ])
    for row in report.get("gpu_nodes") or []:
        lines.append(f"| `{row.get('name')}` | {row.get('count')} |")
    lines.extend([
        "",
        "## Blocker",
        "",
        str(report.get("blocker") or "none"),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _excluded(row: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("id", "project", "signature", "description", "cmd", "cwd", "notes")
    )
    project = str(row.get("project") or "").strip().lower()
    return bool(
        project in {"tmp", "md", "proof", "sched", "python"}
        or "auto-adopted" in text
        or ("adopted" in text and "outside scheduler" in text)
        or "launched outside scheduler" in text
        or "read-only probe" in text
        or "stdout probe" in text
        or "theorem trace probe" in text
    )


def _launched(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    return status in {"running", "done"} and bool(
        row.get("started_at") or row.get("node") or status in {"running", "done"}
    )


def _completed(row: Mapping[str, Any]) -> bool:
    return str(row.get("status") or "").lower() == "done"


def _node(row: Mapping[str, Any]) -> str:
    return str(row.get("node") or row.get("selected_node") or "")


def _is_gpu_node(node: str) -> bool:
    lower = str(node or "").lower()
    return lower.startswith(GPU_NODE_PREFIXES) or "gpu" in lower


def _top(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [
        {"name": str(name), "count": int(count)}
        for name, count in counter.most_common(int(limit))
    ]


def _blocker(
    *,
    strict_completed: list[tuple[Mapping[str, Any], Any]],
    node_counts: Counter[str],
    gpu_node_counts: Counter[str],
    domains: Counter[str],
    unadmitted: list[tuple[Mapping[str, Any], Any]],
    min_nodes: int,
    min_gpu_nodes: int,
    min_completed: int,
    min_workload_domains: int,
    max_unadmitted: int,
) -> str:
    blockers = []
    if len(strict_completed) < int(min_completed):
        blockers.append(f"strict completed count {len(strict_completed)} < {int(min_completed)}")
    if len(node_counts) < int(min_nodes):
        blockers.append(f"node count {len(node_counts)} < {int(min_nodes)}")
    if len(gpu_node_counts) < int(min_gpu_nodes):
        blockers.append(f"gpu node count {len(gpu_node_counts)} < {int(min_gpu_nodes)}")
    if len(domains) < int(min_workload_domains):
        blockers.append(f"workload domains {len(domains)} < {int(min_workload_domains)}")
    if len(unadmitted) > int(max_unadmitted):
        blockers.append(f"strict unadmitted {len(unadmitted)} > {int(max_unadmitted)}")
    return "; ".join(blockers)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_multinode_history_completion_gate(
        min_nodes=args.min_nodes,
        min_gpu_nodes=args.min_gpu_nodes,
        min_completed=args.min_completed,
        min_workload_domains=args.min_workload_domains,
        max_unadmitted=args.max_unadmitted,
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
        prog="python -m algorithm.experiments.multinode_history_completion_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Scheduleurm multi-node history completion gate")
    build.add_argument("--min-nodes", type=int, default=2)
    build.add_argument("--min-gpu-nodes", type=int, default=2)
    build.add_argument("--min-completed", type=int, default=50)
    build.add_argument("--min-workload-domains", type=int, default=3)
    build.add_argument("--max-unadmitted", type=int, default=0)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
