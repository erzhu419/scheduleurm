"""Admission gate for theorem-certified production populations."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.theorem_dispatch.admission import (
    service_domain_certificate,
    task_admission_certificate,
)

from .production_coverage_drilldown import population_label
from .production_load_certificate import (
    classify_record,
    load_scheduler_records,
    _dedupe_records,
    _submitted_at,
    _window_end,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_admission_population_gate(
    *,
    records: Iterable[Mapping[str, Any]],
    window_days: float = 30.0,
    now_ts: float | None = None,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    window_records = [
        dict(row) for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
    ]
    rows = [_admission_row(row) for row in window_records]
    completed_active = [
        row for row in rows
        if bool(row.get("population", {}).get("include_completed_active"))
    ]
    attempted = [
        row for row in rows
        if bool(row.get("population", {}).get("include_attempted"))
    ]
    active_queue = [
        row for row in rows
        if str(row.get("status") or "") in {"queued", "launching", "running"}
    ]
    report = {
        "gate": "service_certified_admission_population",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "window_days": float(window_days),
            "window_s": window_s,
        },
        "completed_active_production": _population_summary(completed_active),
        "attempted_production": _population_summary(attempted),
        "active_queue": _population_summary(active_queue),
        "domain_summary": _domain_summary(rows),
        "strict_classifier_blockers": [
            row for row in completed_active
            if not row.get("strict_service_admitted")
        ][:80],
        "service_domain_blockers": [
            row for row in completed_active
            if not row.get("service_domain_admitted")
        ][:80],
        "active_queue_blockers": [
            row for row in active_queue
            if not row.get("service_domain_admitted")
        ][:80],
        "pass": _population_summary(completed_active)["service_domain_admission_closed"],
        "scope": (
            "the reviewer-facing theorem population is completed-active "
            "production within the fixed window. Raw history and attempted-only "
            "records are reported but are not claimed as the theorem arrival "
            "stream. The live theorem dispatch mode should run with "
            "SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block when future tasks are "
            "to be admitted into the theorem population."
        ),
    }
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    completed = report.get("completed_active_production") or {}
    attempted = report.get("attempted_production") or {}
    active = report.get("active_queue") or {}
    lines = [
        "# Service-Certified Admission Population",
        "",
        "## Summary",
        "",
        "| Population | Records | Strict admitted | Domain admitted | Closed |",
        "|---|---:|---:|---:|---:|",
        _summary_row("completed-active production", completed),
        _summary_row("attempted production", attempted),
        _summary_row("active queue", active),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    blockers = list(report.get("service_domain_blockers") or [])
    if blockers:
        lines.extend([
            "## Completed-Active Service-Domain Blockers",
            "",
            "| Task | Status | Project | Reason | Inferred workload |",
            "|---|---|---|---|---|",
        ])
        for row in blockers[:40]:
            lines.append(
                "| `{task}` | `{status}` | `{project}` | `{reason}` | `{workload}` |".format(
                    task=row.get("task_id"),
                    status=row.get("status"),
                    project=row.get("project"),
                    reason=row.get("admission_reason") or row.get("classification_reason"),
                    workload=row.get("workload_key") or row.get("inferred_workload_key"),
                )
            )
        lines.append("")
    strict_blockers = list(report.get("strict_classifier_blockers") or [])
    if strict_blockers:
        lines.extend([
            "## Strict-Classifier Non-Matches",
            "",
            "These rows are service-domain admitted but do not have a strict",
            "production classifier rule. They should be described as admission",
            "certified, not as strict classifier coverage.",
            "",
            "| Task | Status | Project | Classifier reason | Inferred workload |",
            "|---|---|---|---|---|",
        ])
        for row in strict_blockers[:40]:
            lines.append(
                "| `{task}` | `{status}` | `{project}` | `{reason}` | `{workload}` |".format(
                    task=row.get("task_id"),
                    status=row.get("status"),
                    project=row.get("project"),
                    reason=row.get("classification_reason"),
                    workload=row.get("inferred_workload_key"),
                )
            )
        lines.append("")
    active_blockers = list(report.get("active_queue_blockers") or [])
    if active_blockers:
        lines.extend([
            "## Active-Queue Admission Blockers",
            "",
            "| Task | Status | Project | Reason | Inferred workload |",
            "|---|---|---|---|---|",
        ])
        for row in active_blockers[:40]:
            lines.append(
                "| `{task}` | `{status}` | `{project}` | `{reason}` | `{workload}` |".format(
                    task=row.get("task_id"),
                    status=row.get("status"),
                    project=row.get("project"),
                    reason=row.get("admission_reason") or row.get("classification_reason"),
                    workload=row.get("workload_key") or row.get("inferred_workload_key"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _admission_row(row: Mapping[str, Any]) -> dict[str, Any]:
    pop = population_label(row)
    cls = classify_record(row, include_representative=False)
    workload_key = str(cls.get("workload_key") or "")
    domain = service_domain_certificate(workload_key) if workload_key else {}
    inferred = task_admission_certificate(row)
    strict = (
        workload_key
        and cls.get("mapping_mode") == "strict_measured"
        and bool(domain.get("service_domain_certified"))
    )
    domain_admitted = strict or inferred.admitted
    return {
        "task_id": row.get("id") or row.get("task_id"),
        "status": row.get("status"),
        "project": row.get("project"),
        "signature": row.get("signature"),
        "submitted_at": _submitted_at(row),
        "population": pop,
        "workload_key": workload_key,
        "mapping_mode": cls.get("mapping_mode"),
        "classification_reason": cls.get("reason"),
        "strict_service_admitted": bool(strict),
        "service_domain_admitted": bool(domain_admitted),
        "inferred_workload_key": inferred.workload_key,
        "admission_reason": "" if domain_admitted else (inferred.reason or str(cls.get("reason") or "")),
        "measured_profiles": (
            domain.get("measured_profiles")
            if domain else
            list(inferred.measured_profiles)
        ),
    }


def _population_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    strict = sum(1 for row in rows if row.get("strict_service_admitted"))
    domain = sum(1 for row in rows if row.get("service_domain_admitted"))
    by_reason = Counter(
        str(row.get("admission_reason") or row.get("classification_reason") or "")
        for row in rows if not row.get("strict_service_admitted")
    )
    return {
        "record_count": count,
        "strict_service_admitted_count": strict,
        "service_domain_admitted_count": domain,
        "strict_service_admission_fraction": strict / count if count else 1.0,
        "service_domain_admission_fraction": domain / count if count else 1.0,
        "strict_service_admission_closed": strict == count,
        "service_domain_admission_closed": domain == count,
        "blocker_reasons": dict(by_reason.most_common(20)),
    }


def _domain_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    profiles: dict[str, list[int]] = {}
    for row in rows:
        key = str(row.get("workload_key") or row.get("inferred_workload_key") or "")
        if not key:
            continue
        counts[key] += 1
        if key not in profiles:
            profiles[key] = list(row.get("measured_profiles") or [])
    return {
        "workload_counts": dict(counts.most_common()),
        "profile_domains": profiles,
    }


def _summary_row(name: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {summary.get('record_count', 0)} | "
        f"{summary.get('strict_service_admitted_count', 0)} | "
        f"{summary.get('service_domain_admitted_count', 0)} | "
        f"{str(bool(summary.get('strict_service_admission_closed'))).lower()} |"
    )


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_admission_population_gate(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        ),
        window_days=args.window_days,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.admission_population_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Audit service-certified theorem admission population")
    build.add_argument("--queue-path", default="")
    build.add_argument("--archive-path", default="")
    build.add_argument("--window-days", type=float, default=30.0)
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "admission_population_gate.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "admission_population_gate.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
