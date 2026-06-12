"""Declared finite-domain positive-cover gate.

This gate is deliberately narrower than an arbitrary all-state fabric cover.  It
defines the theorem-facing positive population as the declared finite service
cache domain: each measured workload/profile/node/resource bucket is either a
positive lower-service row, a measured capacity boundary, or an excluded
probe-required row.  Unknown future states remain outside this positive
population until measured.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_declared_finite_domain_positive_cover_gate() -> dict[str, Any]:
    cache = build_default_cache()
    rows: list[dict[str, Any]] = []
    for workload_key in cache.available_workloads():
        records = cache.profiles(workload_key, include_boundaries=True)
        for record in records:
            status = _cover_status(record)
            rows.append({
                "workload_key": record.workload_key,
                "profile": int(record.profile),
                "resource_kind": record.resource_kind,
                "node_bucket": record.node_bucket,
                "command_fingerprint": record.command_fingerprint,
                "unit": record.unit,
                "total_units": float(record.total_units),
                "aggregate_rate": float(record.aggregate_rate),
                "mean_rate": float(record.mean_rate),
                "sample_count": len(record.per_task_rates),
                "capacity_boundary": bool(record.capacity_boundary),
                "source_population": _source_population(record.source),
                "source": record.source,
                "first_seen": "not_available_in_service_cache_v1",
                "last_seen": "not_available_in_service_cache_v1",
                "projection": (
                    "identity_on_exact_measured_profile"
                    if status == "positive_lower_service"
                    else "excluded_from_positive_theorem_population"
                ),
                "rho": 0.0 if status == "positive_lower_service" else None,
                "Lrho": 0.0 if status == "positive_lower_service" else None,
                "cover_status": status,
                "theorem_population": (
                    "positive_declared_finite_domain"
                    if status == "positive_lower_service"
                    else "excluded_or_boundary"
                ),
            })

    status_counts = Counter(row["cover_status"] for row in rows)
    workload_domains = defaultdict(set)
    for row in rows:
        workload_domains[row["workload_key"]].add(int(row["profile"]))
    covered = (
        status_counts.get("positive_lower_service", 0)
        + status_counts.get("capacity_boundary", 0)
        + status_counts.get("probe_required", 0)
    )
    total = len(rows)
    uncovered = status_counts.get("uncovered", 0)
    declared_ready = total > 0 and uncovered == 0 and status_counts.get("positive_lower_service", 0) > 0
    return {
        "gate": "declared_finite_domain_positive_cover_gate",
        "declared_universe": (
            "service_cache workload_key x exact measured profile x node_bucket x "
            "resource_kind x command_fingerprint"
        ),
        "declared_universe_bucket_count": total,
        "workload_domain_count": len(workload_domains),
        "positive_bucket_count": status_counts.get("positive_lower_service", 0),
        "boundary_bucket_count": status_counts.get("capacity_boundary", 0),
        "probe_required_bucket_count": status_counts.get("probe_required", 0),
        "uncovered_bucket_count": uncovered,
        "coverage_fraction": float(covered) / float(total) if total else 0.0,
        "status_counts": dict(status_counts),
        "declared_finite_positive_cover_ready": declared_ready,
        "positive_service_all_state_cover_ready": False,
        "arbitrary_all_state_positive_cover_ready": False,
        "rows": rows,
        "pass": declared_ready,
        "scope": (
            "Closes a declared finite-domain positive cover for exact measured "
            "service-cache buckets.  It does not certify arbitrary future states; "
            "unknown states must enter probe/admission before joining the positive "
            "theorem population."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Declared Finite-Domain Positive-Cover Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `declared_finite_positive_cover_ready` | {str(bool(report.get('declared_finite_positive_cover_ready'))).lower()} |",
        f"| `positive_service_all_state_cover_ready` | {str(bool(report.get('positive_service_all_state_cover_ready'))).lower()} |",
        f"| `declared_universe_bucket_count` | {report.get('declared_universe_bucket_count', 0)} |",
        f"| `workload_domain_count` | {report.get('workload_domain_count', 0)} |",
        f"| `positive_bucket_count` | {report.get('positive_bucket_count', 0)} |",
        f"| `boundary_bucket_count` | {report.get('boundary_bucket_count', 0)} |",
        f"| `probe_required_bucket_count` | {report.get('probe_required_bucket_count', 0)} |",
        f"| `uncovered_bucket_count` | {report.get('uncovered_bucket_count', 0)} |",
        f"| `coverage_fraction` | {float(report.get('coverage_fraction') or 0.0):.6f} |",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, value in sorted((report.get("status_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Declared Universe",
        "",
        str(report.get("declared_universe") or ""),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _cover_status(record: Any) -> str:
    if bool(record.capacity_boundary):
        return "capacity_boundary"
    if float(record.aggregate_rate) > 0.0:
        return "positive_lower_service"
    if record.source or record.command_fingerprint:
        return "probe_required"
    return "uncovered"


def _source_population(source: str) -> str:
    lower = str(source or "").lower()
    if "production" in lower or "completed_history" in lower:
        return "production_or_completed_history_service_cache"
    if "module" in lower or "benchmark" in lower or "local" in lower:
        return "benchmark_or_profile_service_cache"
    return "service_cache"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_declared_finite_domain_positive_cover_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.declared_finite_domain_positive_cover_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build declared finite-domain positive-cover gate")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "declared_finite_domain_positive_cover_gate_20260612.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "declared_finite_domain_positive_cover_gate_20260612.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
