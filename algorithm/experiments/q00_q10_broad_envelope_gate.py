"""q00/q10 broad measured-envelope gate.

The earlier q00/q10 gate closed declared local buckets.  This gate asks a
different question: across every CPU-like workload currently present in the
measured service cache, can future jobs be admitted into a measured lower-
service envelope, while unseen CPU/data-loader shapes are forced to probe?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
Q00_KIND = {"light_control"}
Q10_KIND = {"cpu_heavy", "cpu_sumo_transit"}


def build_q00_q10_broad_envelope_gate() -> dict[str, Any]:
    cache = build_default_cache()
    rows = []
    for key in cache.available_workloads():
        profiles = cache.profiles(key, include_boundaries=True)
        if not profiles:
            continue
        resource_kind = str(profiles[0].resource_kind)
        if resource_kind not in Q00_KIND | Q10_KIND:
            continue
        positive = [
            record for record in profiles
            if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
        ]
        boundaries = [record for record in profiles if record.capacity_boundary]
        node_buckets = sorted({record.node_bucket for record in profiles})
        source_kinds = sorted({_source_kind(record.source, record.node_bucket) for record in profiles})
        rows.append({
            "workload_key": key,
            "quadrant": "q00" if resource_kind in Q00_KIND else "q10",
            "resource_kind": resource_kind,
            "positive_profiles": sorted(int(record.profile) for record in positive),
            "boundary_profiles": sorted(int(record.profile) for record in boundaries),
            "node_buckets": node_buckets,
            "source_kinds": source_kinds,
            "admission_ready": bool(positive),
            "capacity_boundary_ready": bool(boundaries),
            "completed_history_lower_service_only": bool(positive) and not bool(boundaries) and "completed_history" in source_kinds,
            "measured_envelope_ready": bool(positive),
        })
    q00_rows = [row for row in rows if row["quadrant"] == "q00"]
    q10_rows = [row for row in rows if row["quadrant"] == "q10"]
    measured_ready = bool(rows) and all(bool(row["measured_envelope_ready"]) for row in rows)
    boundary_ready_count = sum(1 for row in rows if row["capacity_boundary_ready"])
    completed_history_only_count = sum(1 for row in rows if row["completed_history_lower_service_only"])
    return {
        "gate": "q00_q10_broad_measured_envelope_gate",
        "q00_workload_count": len(q00_rows),
        "q10_workload_count": len(q10_rows),
        "cpu_like_workload_count": len(rows),
        "capacity_boundary_ready_count": boundary_ready_count,
        "completed_history_lower_service_only_count": completed_history_only_count,
        "rows": rows,
        "broad_measured_q00_q10_envelope_ready": measured_ready,
        "broad_all_cpu_dataloader_world_ready": False,
        "pass": measured_ready,
        "admission_rule": (
            "Future q00/q10 tasks are theorem-facing only after inference to one "
            "of these measured workload keys and an exact positive profile; all "
            "unseen CPU/data-loader shapes are probe-required."
        ),
        "scope": (
            "Closes the broad measured q00/q10 envelope currently represented in "
            "the service cache.  It does not claim all possible CPU/data-loader "
            "programs or future hardware regimes are measured."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# q00/q10 Broad Measured-Envelope Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `q00_workload_count` | {report.get('q00_workload_count', 0)} |",
        f"| `q10_workload_count` | {report.get('q10_workload_count', 0)} |",
        f"| `capacity_boundary_ready_count` | {report.get('capacity_boundary_ready_count', 0)} |",
        f"| `completed_history_lower_service_only_count` | {report.get('completed_history_lower_service_only_count', 0)} |",
        f"| `broad_measured_q00_q10_envelope_ready` | {str(bool(report.get('broad_measured_q00_q10_envelope_ready'))).lower()} |",
        f"| `broad_all_cpu_dataloader_world_ready` | {str(bool(report.get('broad_all_cpu_dataloader_world_ready'))).lower()} |",
        "",
        "## Measured CPU-Like Rows",
        "",
        "| Quadrant | Workload | Kind | Positive profiles | Boundary profiles | Sources | Ready |",
        "|---|---|---|---|---|---|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{q}` | `{key}` | `{kind}` | `{pos}` | `{bound}` | `{sources}` | {ready} |".format(
                q=row.get("quadrant"),
                key=row.get("workload_key"),
                kind=row.get("resource_kind"),
                pos=row.get("positive_profiles"),
                bound=row.get("boundary_profiles"),
                sources=row.get("source_kinds"),
                ready=str(bool(row.get("measured_envelope_ready"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _source_kind(source: str, node_bucket: str) -> str:
    text = f"{source} {node_bucket}".lower()
    if "completed_history" in text or "production-history" in text:
        return "completed_history"
    if "protocol:" in text:
        return "protocol"
    if "module" in text or "experiments/runs" in text:
        return "measured_curve"
    return "other"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_q00_q10_broad_envelope_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.q00_q10_broad_envelope_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build q00/q10 broad measured-envelope gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "q00_q10_broad_envelope_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "q00_q10_broad_envelope_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
