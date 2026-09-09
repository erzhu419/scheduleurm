"""Reclassify stable-timeout full-factorial GPU probe reports.

Some co-located real workloads are deliberately run until the profile timeout
instead of stopping at the first stable-rate window.  In that protocol a remote
process can return timeout code 124 even though every controlled child task has
already emitted a task-native stable progress rate.  The full-factorial runner
knows how to interpret those summaries; this utility reapplies that logic to an
existing report so the JSON, cache snapshot, and Markdown artifact stay
reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simulation.service_cache import ServiceRateCache

from .full_factorial_eta_probe_runner import (
    _add_result_to_cache,
    _capacity_boundary_profiles,
    _profile_summary_row,
    markdown_report,
)


def reclassify_report(
    *,
    report_path: Path,
    cache_output: Path | None = None,
    markdown_output: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache = ServiceRateCache()
    rows: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        run_id = str(row.get("run_id") or "")
        profiles = [int(x) for x in (row.get("profiles") or [])]
        profile_rows = [_profile_summary_row(run_id, profile) for profile in profiles]
        capacity_boundary_profiles = sorted(_capacity_boundary_profiles(profile_rows))
        valid = bool(profile_rows) and all(bool(item.get("measurement_valid")) for item in profile_rows)
        refreshed = {
            **dict(row),
            "profile_rows": profile_rows,
            "capacity_boundary_profiles": capacity_boundary_profiles,
            "measurement_valid": valid,
            "reason": "" if valid else "one_or_more_profiles_unstable_or_capacity_boundary",
        }
        rows.append(refreshed)
        _add_result_to_cache(cache, refreshed)

    report["rows"] = rows
    report["admitted_rows"] = [row for row in rows if row.get("measurement_valid")]
    report["boundary_rows"] = [row for row in rows if row.get("capacity_boundary_profiles")]
    report["admitted_count"] = len(report["admitted_rows"])
    report["boundary_count"] = len(report["boundary_rows"])
    report["service_cache_snapshot"] = cache.snapshot()

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cache_output is not None:
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        cache_output.write_text(json.dumps(cache.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.reclassify_stable_timeout_probe")
    parser.add_argument("--report", required=True)
    parser.add_argument("--cache-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    report = reclassify_report(
        report_path=Path(args.report),
        cache_output=Path(args.cache_output) if args.cache_output else None,
        markdown_output=Path(args.markdown_output) if args.markdown_output else None,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "admitted_count": report.get("admitted_count"),
                "boundary_count": report.get("boundary_count"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
