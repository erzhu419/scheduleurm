"""Reclassify CPU full-factorial rows with stable progress before timeout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simulation.service_cache import ServiceRateCache

from .full_factorial_cpu_eta_probe_runner import _add_result_to_cache, markdown_report


def _summary_accepts_stable_timeout(summary: dict[str, Any]) -> bool:
    return (
        bool(summary.get("all_stable_rate_ready"))
        and float(summary.get("aggregate_stable_rate_unit_s") or 0.0) > 0.0
        and str(summary.get("eta_source") or "") == "tqdm/progress"
    )


def _refresh_profile_row(profile_row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(profile_row.get("summary_path") or ""))
    if not path.exists():
        return {**profile_row, "measurement_valid": False, "path_exists": False}
    summary = json.loads(path.read_text(encoding="utf-8"))
    valid = bool(summary.get("measurement_valid")) or _summary_accepts_stable_timeout(summary)
    if valid and not bool(summary.get("measurement_valid")):
        running_count = int(summary.get("running_count") or 1)
        summary["measurement_valid"] = True
        summary["placement_valid"] = True
        summary["returncode_accepted_count"] = running_count
        summary["returncode_acceptance_rule"] = "strict_zero_or_stable_required_rate"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **profile_row,
        "measurement_valid": bool(valid),
        "path_exists": True,
        "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
        "aggregate_stable_rate_unit_s": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
        "eta_source": str(summary.get("eta_source") or "tqdm/progress"),
        "returncode": int(summary.get("returncode") or 0),
    }


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
        profile_rows = [_refresh_profile_row(dict(profile_row)) for profile_row in (row.get("profile_rows") or [])]
        valid = bool(profile_rows) and all(bool(profile_row.get("measurement_valid")) for profile_row in profile_rows)
        refreshed = {
            **dict(row),
            "profile_rows": profile_rows,
            "measurement_valid": valid,
            "reason": "" if valid else "one_or_more_cpu_profiles_missing_stable_progress_rate",
        }
        rows.append(refreshed)
        _add_result_to_cache(cache, refreshed)

    report["rows"] = rows
    report["admitted_rows"] = [row for row in rows if row.get("measurement_valid")]
    report["admitted_count"] = len(report["admitted_rows"])
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
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.reclassify_cpu_stable_timeout_probe")
    parser.add_argument("--report", required=True)
    parser.add_argument("--cache-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    report = reclassify_report(
        report_path=Path(args.report),
        cache_output=Path(args.cache_output) if args.cache_output else None,
        markdown_output=Path(args.markdown_output) if args.markdown_output else None,
    )
    print(json.dumps({"report": str(args.report), "admitted_count": report.get("admitted_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
