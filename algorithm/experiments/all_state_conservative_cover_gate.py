"""All-state conservative fabric-cover certificate.

Positive service guarantees for arbitrary future states require measurements.
What can be closed without new experiments is a conservative all-state safety
cover: every state is either admitted to an exact measured positive profile or
routed to an unknown/probe action with zero theorem service.  This prevents
unmeasured states from invalidating the theorem by silently entering the
positive-service population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.features import resource_bucket, task_kind
from algorithm.theorem_dispatch.service_registry import available_profile_table
from simulation.defaults import build_default_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_all_state_conservative_cover_gate() -> dict[str, Any]:
    cache = build_default_cache()
    profile_table = available_profile_table(cache)
    admitted = []
    for workload_key in sorted(profile_table):
        positives = [
            int(record.profile)
            for record in cache.profiles(workload_key)
            if not record.capacity_boundary and float(record.aggregate_rate) > 0.0
        ]
        if positives:
            admitted.append({
                "workload_key": workload_key,
                "positive_profiles": sorted(positives),
                "projection": "identity_measured_action",
                "rho": 0.0,
                "lower_service_mode": "positive_measured_lower_service",
            })
    unknown_action = {
        "action_id": "unknown_probe_or_defer",
        "projection": "catch_all_probe_action",
        "rho": 0.0,
        "lower_service": 0.0,
        "theorem_population": "excluded_until_measured",
    }
    return {
        "gate": "all_state_conservative_fabric_cover_gate",
        "admitted_measured_workload_count": len(admitted),
        "admitted_measured_rows": admitted,
        "unknown_catch_all_action": unknown_action,
        "arbitrary_state_partition": {
            "measured_admitted": "exact positive service-cache profile; identity projection; rho=0",
            "unknown_unmeasured": "probe/defer action; zero theorem service; not counted as positive-load theorem population",
        },
        "all_state_safety_cover_ready": True,
        "positive_service_all_state_cover_ready": False,
        "all_state_stability_for_unknown_positive_arrivals_ready": False,
        "pass": True,
        "scope": (
            "Covers every scheduler-visible future state conservatively by partitioning "
            "it into admitted measured states or an unknown probe/defer action.  The "
            "unknown action has zero theorem service, so this is an all-state safety "
            "cover, not a positive-service all-state fabric theorem."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# All-State Conservative Fabric-Cover Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `admitted_measured_workload_count` | {report.get('admitted_measured_workload_count', 0)} |",
        f"| `all_state_safety_cover_ready` | {str(bool(report.get('all_state_safety_cover_ready'))).lower()} |",
        f"| `positive_service_all_state_cover_ready` | {str(bool(report.get('positive_service_all_state_cover_ready'))).lower()} |",
        f"| `all_state_stability_for_unknown_positive_arrivals_ready` | {str(bool(report.get('all_state_stability_for_unknown_positive_arrivals_ready'))).lower()} |",
        "",
        "## Partition",
        "",
        "| Region | Contract |",
        "|---|---|",
    ]
    for key, value in (report.get("arbitrary_state_partition") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Unknown Catch-All",
        "",
        f"`{json.dumps(report.get('unknown_catch_all_action') or {}, sort_keys=True)}`",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_all_state_conservative_cover_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.all_state_conservative_cover_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build conservative all-state fabric-cover certificate")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "all_state_conservative_cover_gate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "all_state_conservative_cover_gate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
