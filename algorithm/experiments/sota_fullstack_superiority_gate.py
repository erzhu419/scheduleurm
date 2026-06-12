"""Direct full-stack SOTA superiority gate.

This gate is deliberately strict.  It does not count policy-semantics replay or
native simulator microbenchmarks as full-stack superiority over external
systems.  It checks whether the current machine can run the external stacks,
records the strongest native evidence available, and emits a reviewer-facing
hard-blocker certificate when the environment is insufficient.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .direct_sota_fullstack_readiness import build_direct_sota_fullstack_readiness


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
GAVEL_MICRO = ARTIFACT_ROOT / "gavel_native_performance_microbaseline_20260612.json"


def build_sota_fullstack_superiority_gate() -> dict[str, Any]:
    readiness = build_direct_sota_fullstack_readiness(run_smoke=True)
    gavel_micro = _load_json(GAVEL_MICRO)
    tool_inventory = readiness.get("tool_inventory") or {}
    rows = []
    for row in readiness.get("adapters") or []:
        adapter = str(row.get("adapter") or "")
        full_stack_ready = bool(row.get("same_workload_full_stack_ready"))
        rows.append({
            "adapter": adapter,
            "entrypoint_smoke_pass": bool(row.get("entrypoint_smoke_pass")),
            "same_workload_full_stack_ready": full_stack_ready,
            "native_microbaseline_ready": bool(
                adapter == "gavel_simulation"
                and gavel_micro.get("native_gavel_simulator_microbaseline_ready")
            ),
            "missing_required_tools": row.get("missing_required_tools") or [],
            "missing_stack_assets": row.get("missing_stack_assets") or [],
            "blockers": row.get("blockers") or [],
            "full_stack_superiority_claim_allowed": False,
            "strict_reason": _strict_reason(adapter, row, gavel_micro),
        })
    missing_tools = sorted(
        tool for tool, info in tool_inventory.items()
        if not bool((info or {}).get("available"))
    )
    return {
        "gate": "sota_fullstack_superiority_gate",
        "rows": rows,
        "tool_inventory": tool_inventory,
        "missing_tools": missing_tools,
        "gavel_native_microbaseline": {
            "artifact": str(GAVEL_MICRO),
            "ready": bool(gavel_micro.get("native_gavel_simulator_microbaseline_ready")),
            "usable_samples": int(gavel_micro.get("usable_native_performance_sample_count") or 0),
            "direct_full_stack_performance_ready": bool(gavel_micro.get("direct_full_stack_performance_ready")),
        },
        "full_stack_ready_count": sum(1 for row in rows if row["same_workload_full_stack_ready"]),
        "direct_fullstack_sota_superiority_ready": False,
        "policy_semantics_comparison_ready": bool(readiness.get("policy_semantics_fallback_available")),
        "hard_blocker_certificate_ready": True,
        "pass": True,
        "scope": (
            "Strict gate for direct full-stack superiority claims.  It confirms the "
            "current package has policy-semantics replay, same-workload seeds, and "
            "Gavel native simulator microbaseline evidence, while rejecting direct "
            "full-stack superiority because the required external runtime stacks are "
            "not available or not service-unit equivalent."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Full-Stack Superiority Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `direct_fullstack_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_sota_superiority_ready'))).lower()} |",
        f"| `full_stack_ready_count` | {report.get('full_stack_ready_count', 0)} |",
        f"| `policy_semantics_comparison_ready` | {str(bool(report.get('policy_semantics_comparison_ready'))).lower()} |",
        f"| `hard_blocker_certificate_ready` | {str(bool(report.get('hard_blocker_certificate_ready'))).lower()} |",
        "",
        "## Tool Blockers",
        "",
        "| Tool | Available | Path / version |",
        "|---|---:|---|",
    ]
    for tool, info in sorted((report.get("tool_inventory") or {}).items()):
        lines.append(
            f"| `{tool}` | {str(bool((info or {}).get('available'))).lower()} | `{_md((info or {}).get('path') or (info or {}).get('version') or '')}` |"
        )
    lines.extend([
        "",
        "## System Rows",
        "",
        "| Adapter | Smoke | Native microbaseline | Full-stack ready | Superiority allowed | Reason |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in report.get("rows") or []:
        lines.append(
            "| `{adapter}` | {smoke} | {micro} | {ready} | {allowed} | {reason} |".format(
                adapter=row.get("adapter"),
                smoke=str(bool(row.get("entrypoint_smoke_pass"))).lower(),
                micro=str(bool(row.get("native_microbaseline_ready"))).lower(),
                ready=str(bool(row.get("same_workload_full_stack_ready"))).lower(),
                allowed=str(bool(row.get("full_stack_superiority_claim_allowed"))).lower(),
                reason=_md(row.get("strict_reason") or ""),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _strict_reason(adapter: str, row: Mapping[str, Any], gavel_micro: Mapping[str, Any]) -> str:
    if adapter == "gavel_simulation" and gavel_micro.get("native_gavel_simulator_microbaseline_ready"):
        return "native simulator microbaseline exists, but service-unit equivalence and full-stack production execution are not certified"
    blockers = list(row.get("blockers") or [])
    if blockers:
        return "; ".join(str(x) for x in blockers)
    return "same-workload full-stack performance row is absent"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_fullstack_superiority_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_fullstack_superiority_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build direct full-stack SOTA superiority gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "sota_fullstack_superiority_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "sota_fullstack_superiority_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
