"""System-by-system direct-SOTA claim matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .direct_sota_fullstack_readiness import build_direct_sota_fullstack_readiness


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
ADAPTER_ARTIFACT_ROOT = ARTIFACT_ROOT / "sota_adapters"


SEED_PATTERNS = {
    "gavel_simulation": ("gavel_static_trace_*.json", "gavel_native_trace_*.trace", "gavel_throughputs_*.json"),
    "pollux_adaptdl_scheduler": ("adaptdljobs_*.yaml",),
    "iadeep_kubernetes_extender": ("iadeep_pods_*.yaml",),
    "salus_gpu_sharing": ("salus_benchmark_seed_*.json",),
    "decima_simulator": ("decima_dag_seed_*.json",),
}


def build_sota_fullstack_claim_matrix() -> dict[str, Any]:
    readiness = build_direct_sota_fullstack_readiness(run_smoke=True)
    rows = []
    for row in readiness.get("adapters") or []:
        adapter = str(row.get("adapter") or "")
        seeds = _seed_files(adapter)
        full_stack_ready = bool(row.get("same_workload_full_stack_ready"))
        rows.append({
            "adapter": adapter,
            "repo_exists": bool(row.get("repo_exists")),
            "entrypoint_smoke_pass": bool(row.get("entrypoint_smoke_pass")),
            "native_smoke": row.get("native_smoke") or {},
            "native_performance": row.get("native_performance") or {},
            "seed_file_count": len(seeds),
            "seed_files": seeds[:40],
            "missing_required_tools": row.get("missing_required_tools") or [],
            "missing_stack_assets": row.get("missing_stack_assets") or [],
            "blockers": row.get("blockers") or [],
            "direct_fullstack_same_workload_ready": full_stack_ready,
            "allowed_claim": _allowed_claim(adapter, full_stack_ready),
            "forbidden_claim": _forbidden_claim(adapter, full_stack_ready),
        })
    return {
        "gate": "sota_fullstack_claim_matrix",
        "rows": rows,
        "direct_fullstack_same_workload_ready_count": sum(
            1 for row in rows if row.get("direct_fullstack_same_workload_ready")
        ),
        "all_seed_families_present": all(int(row.get("seed_file_count") or 0) > 0 for row in rows),
        "pass": all(int(row.get("seed_file_count") or 0) > 0 for row in rows),
        "scope": (
            "Claim matrix for external SOTA systems.  It certifies seed readiness "
            "and local smoke status, not full-stack superiority unless an adapter "
            "row has direct_fullstack_same_workload_ready=true."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Full-Stack Claim Matrix",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `all_seed_families_present` | {str(bool(report.get('all_seed_families_present'))).lower()} |",
        f"| `direct_fullstack_same_workload_ready_count` | {report.get('direct_fullstack_same_workload_ready_count', 0)} |",
        "",
        "## System Rows",
        "",
        "| Adapter | Smoke | Seeds | Native microbaseline | Direct full-stack ready | Allowed claim | Forbidden claim |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{adapter}` | {smoke} | {seeds} | {micro} | {ready} | {allowed} | {forbidden} |".format(
                adapter=row.get("adapter"),
                smoke=str(bool(row.get("entrypoint_smoke_pass"))).lower(),
                seeds=int(row.get("seed_file_count") or 0),
                micro=str(bool((row.get("native_performance") or {}).get("native_gavel_simulator_microbaseline_ready"))).lower(),
                ready=str(bool(row.get("direct_fullstack_same_workload_ready"))).lower(),
                allowed=row.get("allowed_claim"),
                forbidden=row.get("forbidden_claim"),
            )
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _seed_files(adapter: str) -> list[str]:
    files = []
    for pattern in SEED_PATTERNS.get(adapter, ()):
        files.extend(str(path) for path in sorted(ADAPTER_ARTIFACT_ROOT.glob(pattern)))
    return files


def _allowed_claim(adapter: str, ready: bool) -> str:
    if ready:
        return "direct same-workload full-stack baseline ready for this artifact row"
    if adapter == "gavel_simulation":
        return "native Gavel trace compatibility and bounded native simulator microbaseline; policy-semantics replay remains the cross-system performance comparison"
    if adapter == "decima_simulator":
        return "Decima seed readiness and entrypoint smoke; scope is Spark-DAG simulator, not GPU co-location baseline"
    return "seed readiness and local entrypoint/tool blocker reporting"


def _forbidden_claim(adapter: str, ready: bool) -> str:
    if ready:
        return "none beyond the exact workload/cluster row"
    return "directly outperforms the external system binary or full stack"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_fullstack_claim_matrix()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_fullstack_claim_matrix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build external SOTA claim matrix")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "sota_fullstack_claim_matrix_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "sota_fullstack_claim_matrix_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
