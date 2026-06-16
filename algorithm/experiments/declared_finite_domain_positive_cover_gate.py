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
import time
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
            source_meta = _source_metadata(record.source)
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
                "capacity_boundary_reason": _boundary_reason(record),
                "boundary_profile": int(record.profile) if record.capacity_boundary else None,
                "first_invalid_profile": int(record.profile) if record.capacity_boundary else None,
                "monotone_exclusion": (
                    f"profiles >= {int(record.profile)} excluded for {record.workload_key}"
                    if record.capacity_boundary else ""
                ),
                "source_population": _source_population(record.source),
                "source": record.source,
                "first_seen": source_meta["first_seen"],
                "last_seen": source_meta["last_seen"],
                "measurement_window_s": source_meta["measurement_window_s"],
                "source_run_ids": source_meta["source_run_ids"],
                "lower_service_method": (
                    "conservative_measured_capacity_boundary"
                    if record.capacity_boundary else "conservative_cache_aggregate_rate"
                ),
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
    classification_fraction = float(covered) / float(total) if total else 0.0
    positive_fraction = float(status_counts.get("positive_lower_service", 0)) / float(total) if total else 0.0
    boundary_fraction = float(status_counts.get("capacity_boundary", 0)) / float(total) if total else 0.0
    uncovered_fraction = float(uncovered) / float(total) if total else 0.0
    source_meta = _aggregate_source_metadata(rows)
    return {
        "gate": "declared_finite_domain_positive_cover_gate",
        "status": "DECLARED_FINITE_POSITIVE_COVER_PASS_ALL_STATE_POSITIVE_FALSE",
        "gate_pass": declared_ready,
        "scoped_claim_ready": declared_ready,
        "strong_claim_ready": False,
        "pass_meaning": (
            "classification and positive-cover certificate over the declared finite "
            "service-cache domain, not arbitrary all-state positive service"
        ),
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
        "declared_domain_classification_fraction": classification_fraction,
        "classification_fraction": classification_fraction,
        "positive_service_fraction": positive_fraction,
        "boundary_fraction": boundary_fraction,
        "uncovered_fraction": uncovered_fraction,
        "status_counts": dict(status_counts),
        "declared_finite_positive_cover_ready": declared_ready,
        "positive_service_all_state_cover_ready": False,
        "arbitrary_all_state_positive_cover_ready": False,
        "service_cache_v2_metadata_ready": source_meta["metadata_ready"],
        "measurement_first_seen": source_meta["measurement_first_seen"],
        "measurement_last_seen": source_meta["measurement_last_seen"],
        "measurement_window_s": source_meta["measurement_window_s"],
        "timestamped_bucket_count": source_meta["timestamped_bucket_count"],
        "untimestamped_bucket_count": source_meta["untimestamped_bucket_count"],
        "source_run_id_count": source_meta["source_run_id_count"],
        "source_run_ids": source_meta["source_run_ids"],
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
        f"| `status` | `{report.get('status')}` |",
        f"| `gate_pass` | {str(bool(report.get('gate_pass'))).lower()} |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `pass_meaning` | {report.get('pass_meaning')} |",
        f"| `declared_finite_positive_cover_ready` | {str(bool(report.get('declared_finite_positive_cover_ready'))).lower()} |",
        f"| `positive_service_all_state_cover_ready` | {str(bool(report.get('positive_service_all_state_cover_ready'))).lower()} |",
        f"| `service_cache_v2_metadata_ready` | {str(bool(report.get('service_cache_v2_metadata_ready'))).lower()} |",
        f"| `declared_universe_bucket_count` | {report.get('declared_universe_bucket_count', 0)} |",
        f"| `workload_domain_count` | {report.get('workload_domain_count', 0)} |",
        f"| `positive_bucket_count` | {report.get('positive_bucket_count', 0)} |",
        f"| `boundary_bucket_count` | {report.get('boundary_bucket_count', 0)} |",
        f"| `probe_required_bucket_count` | {report.get('probe_required_bucket_count', 0)} |",
        f"| `uncovered_bucket_count` | {report.get('uncovered_bucket_count', 0)} |",
        f"| `declared_domain_classification_fraction` | {float(report.get('declared_domain_classification_fraction') or 0.0):.6f} |",
        f"| `positive_service_fraction` | {float(report.get('positive_service_fraction') or 0.0):.6f} |",
        f"| `boundary_fraction` | {float(report.get('boundary_fraction') or 0.0):.6f} |",
        f"| `uncovered_fraction` | {float(report.get('uncovered_fraction') or 0.0):.6f} |",
        f"| `measurement_first_seen` | `{report.get('measurement_first_seen')}` |",
        f"| `measurement_last_seen` | `{report.get('measurement_last_seen')}` |",
        f"| `measurement_window_s` | {report.get('measurement_window_s')} |",
        f"| `timestamped_bucket_count` | {report.get('timestamped_bucket_count', 0)} |",
        f"| `untimestamped_bucket_count` | {report.get('untimestamped_bucket_count', 0)} |",
        f"| `source_run_id_count` | {report.get('source_run_id_count', 0)} |",
        "",
        "This gate passes the scoped certificate. It does not make the adjacent strong claim.",
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


def _source_metadata(source: str) -> dict[str, Any]:
    path = Path(str(source or "")).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if path.exists():
        mtime = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
        run_id = path.parent.name or path.stem
        return {
            "first_seen": mtime,
            "last_seen": mtime,
            "measurement_window_s": 0.0,
            "source_run_ids": [run_id],
        }
    return {
        "first_seen": "not_available_in_service_cache_v1",
        "last_seen": "not_available_in_service_cache_v1",
        "measurement_window_s": None,
        "source_run_ids": [],
    }


def _aggregate_source_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_seen = [
        str(row.get("first_seen"))
        for row in rows
        if row.get("first_seen")
        and row.get("first_seen") != "not_available_in_service_cache_v1"
    ]
    last_seen = [
        str(row.get("last_seen"))
        for row in rows
        if row.get("last_seen")
        and row.get("last_seen") != "not_available_in_service_cache_v1"
    ]
    source_run_ids = sorted({
        str(run_id)
        for row in rows
        for run_id in (row.get("source_run_ids") or [])
        if str(run_id)
    })
    min_first = min(first_seen) if first_seen else "not_available_in_service_cache_v1"
    max_last = max(last_seen) if last_seen else "not_available_in_service_cache_v1"
    window_s = _timestamp_window_s(min_first, max_last)
    timestamped = len(first_seen)
    total = len(rows)
    return {
        "metadata_ready": timestamped > 0 and bool(source_run_ids),
        "measurement_first_seen": min_first,
        "measurement_last_seen": max_last,
        "measurement_window_s": window_s,
        "timestamped_bucket_count": timestamped,
        "untimestamped_bucket_count": max(total - timestamped, 0),
        "source_run_id_count": len(source_run_ids),
        "source_run_ids": source_run_ids,
    }


def _timestamp_window_s(first_seen: str, last_seen: str) -> float | None:
    if "not_available" in first_seen or "not_available" in last_seen:
        return None
    try:
        first = time.strptime(first_seen, "%Y-%m-%dT%H:%M:%S%z")
        last = time.strptime(last_seen, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None
    return max(time.mktime(last) - time.mktime(first), 0.0)


def _boundary_reason(record: Any) -> str:
    if not bool(record.capacity_boundary):
        return ""
    source = str(getattr(record, "source", "") or "").lower()
    if "oom" in source or "memory" in source:
        return "oom_or_memory_boundary"
    if "timeout" in source:
        return "timeout_boundary"
    if float(getattr(record, "aggregate_rate", 0.0) or 0.0) <= 0.0:
        return "nonpositive_rate_or_capacity_boundary"
    return "measured_capacity_boundary"


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
