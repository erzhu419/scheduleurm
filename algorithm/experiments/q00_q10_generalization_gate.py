"""q00/q10 local-bucket and broader-generalization evidence gate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_q00_q10_generalization_gate() -> dict[str, Any]:
    cache = build_default_cache()
    workloads = sorted(cache.available_workloads())
    rows = []
    for key in workloads:
        profiles = [record.snapshot() for record in cache.profiles(key, include_boundaries=True)]
        if not profiles:
            continue
        resource_kind = str(profiles[0].get("resource_kind") or "")
        if resource_kind not in {"light_control", "cpu_heavy", "cpu_sumo_transit"}:
            continue
        positive = [
            row for row in profiles
            if not row.get("capacity_boundary") and float(row.get("aggregate_rate") or 0.0) > 0.0
        ]
        boundaries = [row for row in profiles if row.get("capacity_boundary")]
        rows.append({
            "workload_key": key,
            "resource_kind": resource_kind,
            "node_buckets": sorted({str(row.get("node_bucket") or "") for row in profiles}),
            "positive_profile_count": len(positive),
            "positive_profiles": sorted(int(row.get("profile") or 0) for row in positive),
            "boundary_profiles": sorted(int(row.get("profile") or 0) for row in boundaries),
            "max_positive_profile": max((int(row.get("profile") or 0) for row in positive), default=0),
            "source_kinds": _source_kinds(profiles),
            "theorem_grade_curve": _theorem_grade_curve(positive, boundaries),
        })
    q00_local = _row_by_key(rows, "light_control_local")
    q10_local = _row_by_key(rows, "cpu_heavy_local_bench")
    remote_cpu_rows = [
        row for row in rows
        if row["resource_kind"] in {"cpu_heavy", "cpu_sumo_transit"}
        and any("jtl110cpu" in bucket or "production-history" in bucket for bucket in row["node_buckets"])
    ]
    multi_profile_remote = [
        row for row in remote_cpu_rows
        if row["positive_profile_count"] >= 2
    ]
    return {
        "gate": "q00_q10_generalization_gate",
        "q00_local_bucket": q00_local,
        "q10_local_bucket": q10_local,
        "remote_cpu_evidence_rows": remote_cpu_rows,
        "remote_cpu_multi_profile_rows": multi_profile_remote,
        "q00_declared_local_bucket_closed": _closed_local(q00_local, min_positive=10),
        "q10_declared_local_bucket_closed": _closed_local(q10_local, min_positive=8),
        "remote_cpu_has_multi_profile_evidence": bool(multi_profile_remote),
        "remote_cpu_broad_theorem_ready": _remote_broad_ready(multi_profile_remote),
        "broad_q00_q10_generalization_ready": False,
        "pass": _closed_local(q00_local, min_positive=10)
        and _closed_local(q10_local, min_positive=8)
        and bool(remote_cpu_rows),
        "scope": (
            "Evidence gate for q00/q10 scope. Declared local q00/q10 buckets are closed; "
            "remote CPU/SUMO production evidence is inventoried and includes multi-profile "
            "remote CPU data where present. Broad all-CPU/data-loader generalization remains "
            "false until remote curves include comparable capacity boundaries and admission rules."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# q00/q10 Generalization Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `q00_declared_local_bucket_closed` | {str(bool(report.get('q00_declared_local_bucket_closed'))).lower()} |",
        f"| `q10_declared_local_bucket_closed` | {str(bool(report.get('q10_declared_local_bucket_closed'))).lower()} |",
        f"| `remote_cpu_has_multi_profile_evidence` | {str(bool(report.get('remote_cpu_has_multi_profile_evidence'))).lower()} |",
        f"| `remote_cpu_broad_theorem_ready` | {str(bool(report.get('remote_cpu_broad_theorem_ready'))).lower()} |",
        f"| `broad_q00_q10_generalization_ready` | {str(bool(report.get('broad_q00_q10_generalization_ready'))).lower()} |",
        "",
        "## Local Buckets",
        "",
        "| Bucket | Positive profiles | Boundary profiles | Node buckets |",
        "|---|---|---|---|",
    ]
    for label, key in (("q00", "q00_local_bucket"), ("q10", "q10_local_bucket")):
        row = report.get(key) or {}
        lines.append(
            f"| `{label}:{row.get('workload_key')}` | `{row.get('positive_profiles')}` | "
            f"`{row.get('boundary_profiles')}` | `{row.get('node_buckets')}` |"
        )
    lines.extend([
        "",
        "## Remote CPU Evidence",
        "",
        "| Workload | Kind | Node buckets | Positive profiles | Boundary profiles | Source kinds |",
        "|---|---|---|---|---|---|",
    ])
    for row in report.get("remote_cpu_evidence_rows") or []:
        lines.append(
            f"| `{row.get('workload_key')}` | `{row.get('resource_kind')}` | "
            f"`{row.get('node_buckets')}` | `{row.get('positive_profiles')}` | "
            f"`{row.get('boundary_profiles')}` | `{row.get('source_kinds')}` |"
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _source_kinds(profiles: list[Mapping[str, Any]]) -> list[str]:
    kinds = set()
    for row in profiles:
        source = str(row.get("source") or "")
        if source.startswith("protocol:"):
            kinds.add("protocol")
        elif "completed_history" in source or "production-history" in str(row.get("node_bucket") or ""):
            kinds.add("completed_history")
        elif "module" in source or "experiments/runs" in source:
            kinds.add("measured_curve")
        else:
            kinds.add("other")
    return sorted(kinds)


def _theorem_grade_curve(positive: list[Mapping[str, Any]], boundaries: list[Mapping[str, Any]]) -> bool:
    return bool(positive) and bool(boundaries) and len(positive) >= 2


def _row_by_key(rows: list[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("workload_key") == key:
            return row
    return {}


def _closed_local(row: Mapping[str, Any], *, min_positive: int) -> bool:
    return (
        bool(row)
        and int(row.get("positive_profile_count") or 0) >= int(min_positive)
        and bool(row.get("boundary_profiles"))
        and bool(row.get("theorem_grade_curve"))
    )


def _remote_broad_ready(rows: list[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    return any(bool(row.get("theorem_grade_curve")) for row in rows)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_q00_q10_generalization_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.q00_q10_generalization_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build q00/q10 generalization evidence gate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "q00_q10_generalization_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "q00_q10_generalization_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
