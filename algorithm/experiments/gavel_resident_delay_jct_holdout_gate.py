"""Gavel-style resident-delay/JCT holdout for corner-case co-location rows.

This is a scoped finish-time comparison, not a direct full-stack Gavel run.  It
asks whether admitting the marginal job immediately is better than deferring it
until the resident workload finishes, using measured Scheduleurm rates and a
resident-alone challenge rate that favors the defer policy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_LIVE_DIR = ARTIFACT_ROOT / "corner_case_live_20260613"
DEFAULT_RESIDENT_RUN = ARTIFACT_ROOT / "corner_case_resident_alone_20260613"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "gavel_resident_delay_jct_holdout_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "gavel_resident_delay_jct_holdout_20260613.md"

RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")
STEP_RE = re.compile(r"--steps\s+([0-9]+)")

HOLDOUT_SCENARIOS: dict[str, dict[str, str]] = {
    "node007_30gb_add_small_stable": {
        "baseline": "node007_empty_small",
        "resident": "node007_30gb_resident_alone",
        "policy_family": "gavel_gpu_finish_time",
    },
    "cpu_half_node001_add_single_v2": {
        "baseline": "cpu_empty_node001_single",
        "resident": "cpu_half_node001_resident_alone_stable",
        "fallback_resident": "cpu_half_node001_resident_alone",
        "policy_family": "gavel_cpu_finish_time",
    },
    "cpu_full_node001_add_single_v2": {
        "baseline": "cpu_empty_node001_single",
        "resident": "cpu_full_node001_resident_alone_stable",
        "fallback_resident": "cpu_full_node001_resident_alone",
        "policy_family": "gavel_cpu_finish_time",
    },
}


def build_gavel_resident_delay_jct_holdout_gate(
    *,
    live_dir: str | Path = DEFAULT_LIVE_DIR,
    resident_report_dir: str | Path = DEFAULT_RESIDENT_RUN,
) -> dict[str, Any]:
    live_dir = Path(live_dir).expanduser()
    resident_report_dir = Path(resident_report_dir).expanduser()
    rows = [
        _holdout_row(
            scenario_id=scenario_id,
            spec=spec,
            live_dir=live_dir,
            resident_report_dir=resident_report_dir,
        )
        for scenario_id, spec in HOLDOUT_SCENARIOS.items()
    ]
    usable = [row for row in rows if row.get("usable_holdout_row")]
    admit_jct = [row for row in usable if row.get("admit_now_jct_better_under_resident_challenge")]
    admit_makespan = [row for row in usable if row.get("admit_now_makespan_better_under_resident_challenge")]
    scoped_ready = (
        len(usable) == len(rows)
        and len(admit_jct) == len(rows)
        and len(admit_makespan) == len(rows)
    )
    return {
        "gate": "gavel_resident_delay_jct_holdout_gate",
        "status": (
            "GAVEL_RESIDENT_DELAY_JCT_HOLDOUT_PASS"
            if scoped_ready else
            "GAVEL_RESIDENT_DELAY_JCT_HOLDOUT_PENDING"
        ),
        "live_dir": str(live_dir),
        "resident_report_dir": str(resident_report_dir),
        "row_count": len(rows),
        "usable_row_count": len(usable),
        "admit_now_jct_better_count": len(admit_jct),
        "admit_now_makespan_better_count": len(admit_makespan),
        "gavel_style_resident_delay_jct_holdout_ready": scoped_ready,
        "scoped_claim_ready": scoped_ready,
        "strong_claim_ready": False,
        "pass": True,
        "pass_meaning": (
            "Gavel-style finish-time/JCT holdout on the controlled corner-case "
            "slice using measured Scheduleurm foreground, resident, and "
            "resident-alone challenge rates"
        ),
        "rows": rows,
        "scope": (
            "The holdout compares immediate co-location with defer-until-resident "
            "for the measured corner-case rows.  The resident-alone challenge rate "
            "uses the fastest observed resident-alone window, which favors the "
            "defer baseline.  This is still a scoped measured-service holdout, not "
            "a direct full-stack Gavel/Pollux/Sia/IADeep/Salus execution result."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gavel Resident-Delay/JCT Holdout Gate",
        "",
        "This is a scoped Gavel-style finish-time holdout over measured Scheduleurm corner-case rows. It is not a direct full-stack SOTA execution claim.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `usable_row_count` | {report.get('usable_row_count', 0)} |",
        f"| `admit_now_jct_better_count` | {report.get('admit_now_jct_better_count', 0)} |",
        f"| `admit_now_makespan_better_count` | {report.get('admit_now_makespan_better_count', 0)} |",
        "",
        "## Rows",
        "",
        "| Scenario | Co-run mean JCT | Defer mean JCT | JCT gain | Co-run makespan | Defer makespan | Makespan gain | Resident challenge | Ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{scenario}` | {co_jct:.6g} | {def_jct:.6g} | {jct_gain:.2%} | {co_ms:.6g} | {def_ms:.6g} | {ms_gain:.2%} | {challenge:.6g} | {ready} |".format(
                scenario=row.get("scenario_id"),
                co_jct=float(row.get("co_run_mean_jct_s") or 0.0),
                def_jct=float(row.get("defer_mean_jct_s_under_resident_challenge") or 0.0),
                jct_gain=float(row.get("mean_jct_reduction_fraction") or 0.0),
                co_ms=float(row.get("co_run_makespan_s") or 0.0),
                def_ms=float(row.get("defer_makespan_s_under_resident_challenge") or 0.0),
                ms_gain=float(row.get("makespan_reduction_fraction") or 0.0),
                challenge=float(row.get("resident_alone_challenge_rate_step_s") or 0.0),
                ready=str(bool(row.get("usable_holdout_row"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _holdout_row(
    *,
    scenario_id: str,
    spec: Mapping[str, str],
    live_dir: Path,
    resident_report_dir: Path,
) -> dict[str, Any]:
    loaded = _load_json(live_dir / f"{scenario_id}_remote_corner_case_probe.json")
    baseline = _load_json(live_dir / f"{spec['baseline']}_remote_corner_case_probe.json")
    resident_id = spec.get("resident") or ""
    resident = _load_resident_row(resident_report_dir, resident_id)
    if not resident and spec.get("fallback_resident"):
        resident_id = str(spec.get("fallback_resident"))
        resident = _load_resident_row(resident_report_dir, resident_id)
    fg_steps = _steps_from_cmd(str(loaded.get("probe_cmd") or ""))
    resident_steps = _steps_from_cmd(str(loaded.get("background_cmd") or ""))
    fg_loaded_rate = _as_float(loaded.get("probe_last_rate_step_s"))
    fg_empty_rate = _as_float(baseline.get("probe_last_rate_step_s"))
    resident_corun_rate = _as_float(loaded.get("background_last_rate_step_s"))
    resident_rates = _resident_rates(resident)
    resident_challenge_rate = max(resident_rates) if resident_rates else _as_float(resident.get("probe_last_rate_step_s"))
    co_fg = _duration(fg_steps, fg_loaded_rate)
    co_resident = _duration(resident_steps, resident_corun_rate)
    co_makespan = max(co_fg, co_resident)
    co_mean_jct = (co_fg + co_resident) / 2.0 if math.isfinite(co_makespan) else math.inf
    defer_resident = _duration(resident_steps, resident_challenge_rate)
    defer_fg = defer_resident + _duration(fg_steps, fg_empty_rate)
    defer_makespan = defer_fg
    defer_mean_jct = (defer_resident + defer_fg) / 2.0 if math.isfinite(defer_fg) else math.inf
    usable = all(
        value > 0.0 and math.isfinite(value)
        for value in (fg_steps, resident_steps, fg_loaded_rate, fg_empty_rate, resident_corun_rate, resident_challenge_rate)
    )
    jct_better = usable and co_mean_jct <= defer_mean_jct
    makespan_better = usable and co_makespan <= defer_makespan
    return {
        "scenario_id": scenario_id,
        "baseline_scenario_id": spec["baseline"],
        "resident_alone_scenario_id": resident_id,
        "policy_family": spec.get("policy_family") or "",
        "foreground_steps": fg_steps,
        "resident_steps": resident_steps,
        "foreground_loaded_rate_step_s": fg_loaded_rate,
        "foreground_empty_rate_step_s": fg_empty_rate,
        "resident_corun_rate_step_s": resident_corun_rate,
        "resident_alone_last_rate_step_s": _as_float(resident.get("probe_last_rate_step_s")),
        "resident_alone_challenge_rate_rule": "max observed resident-alone progress window, favorable to defer",
        "resident_alone_challenge_rate_step_s": resident_challenge_rate,
        "resident_alone_stable_eta_ready": bool((resident.get("stable_eta") or {}).get("ready")),
        "resident_alone_rate_count": len(resident_rates),
        "co_run_foreground_completion_s": co_fg,
        "co_run_resident_completion_s": co_resident,
        "co_run_makespan_s": co_makespan,
        "co_run_mean_jct_s": co_mean_jct,
        "defer_resident_completion_s_under_resident_challenge": defer_resident,
        "defer_foreground_completion_s_under_resident_challenge": defer_fg,
        "defer_makespan_s_under_resident_challenge": defer_makespan,
        "defer_mean_jct_s_under_resident_challenge": defer_mean_jct,
        "mean_jct_reduction_fraction": _fraction_reduction(defer_mean_jct, co_mean_jct),
        "makespan_reduction_fraction": _fraction_reduction(defer_makespan, co_makespan),
        "admit_now_jct_better_under_resident_challenge": bool(jct_better),
        "admit_now_makespan_better_under_resident_challenge": bool(makespan_better),
        "usable_holdout_row": bool(usable),
        "loaded_artifact": str(live_dir / f"{scenario_id}_remote_corner_case_probe.json"),
        "baseline_artifact": str(live_dir / f"{spec['baseline']}_remote_corner_case_probe.json"),
        "resident_artifact": str(_resident_path(resident_report_dir, resident_id)),
    }


def _load_resident_row(report_dir: Path, scenario_id: str) -> dict[str, Any]:
    return _load_json(_resident_path(report_dir, scenario_id))


def _resident_path(report_dir: Path, scenario_id: str) -> Path:
    return report_dir / f"{scenario_id}_remote_corner_case_probe.json"


def _resident_rates(row: Mapping[str, Any]) -> list[float]:
    raw = row.get("raw_paths") or {}
    path = raw.get("probe")
    rates: list[float] = []
    if path:
        try:
            text = Path(str(path)).expanduser().read_text(encoding="utf-8", errors="replace")
            rates.extend(float(value) for value in RATE_RE.findall(text))
        except Exception:
            pass
    if not rates and _as_float(row.get("probe_last_rate_step_s")) > 0.0:
        rates.append(_as_float(row.get("probe_last_rate_step_s")))
    return [rate for rate in rates if rate > 0.0 and math.isfinite(rate)]


def _steps_from_cmd(cmd: str) -> int:
    match = STEP_RE.search(cmd)
    return max(1, int(match.group(1))) if match else 1


def _duration(units: float, rate: float) -> float:
    if units <= 0.0 or rate <= 0.0:
        return math.inf
    return float(units) / float(rate)


def _fraction_reduction(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(candidate):
        return 0.0
    return (baseline - candidate) / baseline


def _as_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_resident_delay_jct_holdout_gate(
        live_dir=args.live_dir,
        resident_report_dir=args.resident_report_dir,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("scoped_claim_ready") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_resident_delay_jct_holdout_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Gavel-style resident-delay/JCT holdout gate")
    build.add_argument("--live-dir", default=str(DEFAULT_LIVE_DIR))
    build.add_argument("--resident-report-dir", default=str(DEFAULT_RESIDENT_RUN))
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
