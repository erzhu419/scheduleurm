"""Collect supplemental selected-profile rates for the stochastic LCB gate."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import records_from_summary_file


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "selected_profile_holdout_supplemental_samples_20260612.json"


def build_supplemental_samples(
    *,
    summary_glob: str,
    workload_key: str,
    profile: int,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str,
    context: str = "selected",
    existing_path: str | Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    existing = _load_json(existing_path) if existing_path else {}
    rows = list(existing.get("rows") or [])
    seen_keys = {
        (
            str(row.get("source") or ""),
            str(row.get("workload_key") or ""),
            int(row.get("profile") or 0),
            str(row.get("context") or "selected"),
        )
        for row in rows
    }
    added = []
    for path in sorted(glob.glob(str(Path(summary_glob).expanduser()))):
        source = str(Path(path).expanduser())
        key = (source, workload_key, int(profile), str(context or "selected"))
        if key in seen_keys:
            continue
        for record in records_from_summary_file(
            Path(source),
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
        ):
            if int(record.profile) != int(profile):
                continue
            rates = [float(x) for x in record.per_task_rates if float(x) > 0.0]
            if not rates or record.capacity_boundary:
                continue
            row = {
                "workload_key": workload_key,
                "profile": int(profile),
                "context": context,
                "command_fingerprint": command_fingerprint,
                "resource_kind": resource_kind,
                "total_units": float(total_units),
                "node_bucket": node_bucket,
                "source": source,
                "rates_per_task": rates,
                "rate_count": len(rates),
                "aggregate_rate": float(record.aggregate_rate),
                "unit": record.unit,
            }
            rows.append(row)
            added.append(row)
            seen_keys.add(key)
    return {
        "gate": "selected_profile_holdout_supplemental_samples",
        "rows": rows,
        "sources": sorted({str(row.get("source") or "") for row in rows}),
        "row_count": len(rows),
        "rate_count": sum(len(row.get("rates_per_task") or []) for row in rows),
        "added_row_count": len(added),
        "added_rate_count": sum(len(row.get("rates_per_task") or []) for row in added),
        "pass": True,
        "scope": (
            "Supplemental raw per-task service rates for selected-profile stochastic "
            "LCB calibration.  These rows are ignored unless the selected-profile "
            "LCB gate is invoked with this supplemental path."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Selected-Profile Holdout Supplemental Samples",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `row_count` | {report.get('row_count', 0)} |",
        f"| `rate_count` | {report.get('rate_count', 0)} |",
        f"| `added_row_count` | {report.get('added_row_count', 0)} |",
        f"| `added_rate_count` | {report.get('added_rate_count', 0)} |",
        "",
        "| Workload | Profile | Context | Rates | Source |",
        "|---|---:|---|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{workload}` | {profile} | `{context}` | {rates} | `{source}` |".format(
                workload=row.get("workload_key"),
                profile=row.get("profile"),
                context=row.get("context"),
                rates=len(row.get("rates_per_task") or []),
                source=str(row.get("source") or ""),
            )
        )
    return "\n".join(lines) + "\n"


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_add(args: argparse.Namespace) -> int:
    report = build_supplemental_samples(
        summary_glob=args.summary_glob,
        workload_key=args.workload_key,
        profile=args.profile,
        command_fingerprint=args.command_fingerprint,
        resource_kind=args.resource_kind,
        total_units=args.total_units,
        node_bucket=args.node_bucket,
        context=args.context,
        existing_path=args.output if args.append else None,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.selected_profile_holdout_supplemental_samples")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="Add summary rates to selected-profile supplemental sample file")
    add.add_argument("--summary-glob", required=True)
    add.add_argument("--workload-key", required=True)
    add.add_argument("--profile", type=int, required=True)
    add.add_argument("--command-fingerprint", required=True)
    add.add_argument("--resource-kind", required=True)
    add.add_argument("--total-units", type=float, required=True)
    add.add_argument("--node-bucket", required=True)
    add.add_argument("--context", default="selected")
    add.add_argument("--append", action="store_true")
    add.add_argument("--output", default=str(DEFAULT_OUTPUT))
    add.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "selected_profile_holdout_supplemental_samples_20260612.md"))
    add.set_defaults(func=_cmd_add)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
