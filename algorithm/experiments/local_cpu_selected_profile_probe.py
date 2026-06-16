"""Direct local CPU selected-profile probe.

This collector is intentionally outside the live scheduler.  It measures a
bounded local CPU co-location profile and writes a summary compatible with the
service-cache supplemental-sample importer.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")


def build_local_cpu_selected_profile_probe(
    *,
    run_id: str,
    profile: int = 8,
    mode: str = "cpu",
    steps: int = 1000000,
    work_items: int = 1000000,
    sleep_s: float = 0.0,
    measure_s: int = 45,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    phase = f"profile_{int(profile)}_per_resource"
    script = REPO_ROOT / "algorithm" / "experiments" / "cpu_progress_benchmark.py"
    procs: list[subprocess.Popen[str]] = []
    started_at = time.time()
    try:
        for idx in range(max(1, int(profile))):
            label = f"{run_id}-{phase}-{idx}"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--steps",
                    str(int(steps)),
                    "--mode",
                    mode,
                    "--work-items",
                    str(int(work_items)),
                    "--sleep-s",
                    f"{float(sleep_s):.6f}",
                    "--label",
                    label,
                ],
                cwd=str(REPO_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append(proc)
        deadline = time.time() + max(1, int(measure_s))
        while time.time() < deadline and any(proc.poll() is None for proc in procs):
            time.sleep(max(0.2, float(poll_s)))
    finally:
        outputs = []
        for idx, proc in enumerate(procs):
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except OSError:
                    pass
            try:
                out, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate(timeout=5)
            path = raw_dir / f"{phase}_{idx}.log"
            path.write_text(out or "", encoding="utf-8")
            outputs.append(out or "")
    elapsed = max(0.001, time.time() - started_at)
    rates = [_last_rate(output) for output in outputs]
    rates = [rate for rate in rates if rate > 0.0]
    summary = {
        "phase": phase,
        "probe": "local_cpu_selected_profile_probe",
        "run_id": run_id,
        "mode": mode,
        "steps": int(steps),
        "work_items": int(work_items),
        "sleep_s": float(sleep_s),
        "measure_s": int(measure_s),
        "elapsed_wall_s": elapsed,
        "running_count": int(profile),
        "running_with_rate_count": len(rates),
        "rates_unit_s": rates,
        "rate_units": ["unit"],
        "aggregate_active_rate_unit_s": sum(rates),
        "mean_active_rate_unit_s": mean(rates) if rates else 0.0,
        "measurement_aggregate_rate_unit_s_median": sum(rates),
        "blocked_count": 0,
        "eviction_count": 0,
        "placement_valid": len(rates) == int(profile),
        "placement_issues": [] if len(rates) == int(profile) else [
            f"running_with_rate_count {len(rates)} != expected {int(profile)}"
        ],
        "status_counts": {"sampled": len(rates), "missing_rate": max(0, int(profile) - len(rates))},
        "capacity_boundary": False,
    }
    summary_path = report_dir / f"{phase}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = report_dir / f"{phase}_summary.md"
    markdown_path.write_text(_markdown(summary, summary_path), encoding="utf-8")
    result = {
        "gate": "local_cpu_selected_profile_probe",
        "run_id": run_id,
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "profile": int(profile),
        "rate_count": len(rates),
        "aggregate_active_rate_unit_s": sum(rates),
        "pass": len(rates) == int(profile),
    }
    (report_dir / "probe_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _last_rate(output: str) -> float:
    matches = RATE_RE.findall(output or "")
    if not matches:
        return 0.0
    return float(matches[-1])


def _markdown(summary: Mapping[str, Any], summary_path: Path) -> str:
    return "\n".join([
        "# Local CPU Selected-Profile Probe",
        "",
        f"Summary: `{summary_path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `phase` | `{summary.get('phase')}` |",
        f"| `running_count` | {summary.get('running_count')} |",
        f"| `running_with_rate_count` | {summary.get('running_with_rate_count')} |",
        f"| `aggregate_active_rate_unit_s` | {float(summary.get('aggregate_active_rate_unit_s') or 0.0):.6g} |",
        f"| `mean_active_rate_unit_s` | {float(summary.get('mean_active_rate_unit_s') or 0.0):.6g} |",
        f"| `placement_valid` | {str(bool(summary.get('placement_valid'))).lower()} |",
        "",
    ])


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_local_cpu_selected_profile_probe(
        run_id=args.run_id,
        profile=args.profile,
        mode=args.mode,
        steps=args.steps,
        work_items=args.work_items,
        sleep_s=args.sleep_s,
        measure_s=args.measure_s,
        poll_s=args.poll_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.local_cpu_selected_profile_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Measure a bounded local CPU selected profile")
    build.add_argument("--run-id", required=True)
    build.add_argument("--profile", type=int, default=8)
    build.add_argument("--mode", choices=("light", "cpu"), default="cpu")
    build.add_argument("--steps", type=int, default=1000000)
    build.add_argument("--work-items", type=int, default=1000000)
    build.add_argument("--sleep-s", type=float, default=0.0)
    build.add_argument("--measure-s", type=int, default=45)
    build.add_argument("--poll-s", type=float, default=2.0)
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
