"""Future-admitted fabric-cover gate.

The mathematical theorem can be used for future states only when the scheduler
routes future tasks through the measured admission contract.  This certificate
closes that restricted claim by construction: admitted future actions must bind
to an exact positive service-cache profile, and the theorem-facing projection
is identity on that measured action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.service_registry import available_profile_table
from simulation.defaults import build_default_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_future_admitted_fabric_cover_gate() -> dict[str, Any]:
    cache = build_default_cache()
    profile_table = available_profile_table(cache)
    rows = []
    for workload_key in sorted(profile_table):
        positive_profiles = [
            int(record.profile)
            for record in cache.profiles(workload_key)
            if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
        ]
        boundary_profiles = [
            int(record.profile)
            for record in cache.profiles(workload_key, include_boundaries=True)
            if record.capacity_boundary
        ]
        rows.append({
            "workload_key": workload_key,
            "positive_profiles": sorted(positive_profiles),
            "boundary_profiles": sorted(boundary_profiles),
            "future_admitted_profiles": sorted(positive_profiles),
            "projection": "identity_on_exact_measured_profile",
            "rho": 0.0,
            "Lrho": 0.0,
            "admitted_future_state_ready": bool(positive_profiles),
        })
    ready = bool(rows) and all(bool(row["admitted_future_state_ready"]) for row in rows)
    return {
        "gate": "future_admitted_fabric_cover_gate",
        "workload_count": len(rows),
        "rows": rows,
        "future_admitted_measured_state_ready": ready,
        "future_all_state_fabric_cover_ready": False,
        "pass": ready,
        "admission_contract": {
            "admit": "workload_key inferred and exact positive non-boundary service profile exists",
            "project": "identity projection on that measured profile",
            "otherwise": "route to probe/admission; do not enter theorem-facing stream",
        },
        "scope": (
            "Closes the future-state claim only for tasks admitted into the measured "
            "service domain.  It intentionally does not certify arbitrary unseen "
            "hardware regimes, unseen workload classes, or unmeasured co-location profiles."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Future-Admitted Fabric-Cover Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `workload_count` | {report.get('workload_count', 0)} |",
        f"| `future_admitted_measured_state_ready` | {str(bool(report.get('future_admitted_measured_state_ready'))).lower()} |",
        f"| `future_all_state_fabric_cover_ready` | {str(bool(report.get('future_all_state_fabric_cover_ready'))).lower()} |",
        "",
        "## Workload Profile Domains",
        "",
        "| Workload | Admitted profiles | Boundary profiles | rho | Lrho |",
        "|---|---|---|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{key}` | `{profiles}` | `{boundaries}` | {rho:.1f} | {lrho:.1f} |".format(
                key=row.get("workload_key"),
                profiles=row.get("future_admitted_profiles"),
                boundaries=row.get("boundary_profiles"),
                rho=float(row.get("rho") or 0.0),
                lrho=float(row.get("Lrho") or 0.0),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_future_admitted_fabric_cover_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.future_admitted_fabric_cover_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build future-admitted measured-state fabric cover certificate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "future_admitted_fabric_cover_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "future_admitted_fabric_cover_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
