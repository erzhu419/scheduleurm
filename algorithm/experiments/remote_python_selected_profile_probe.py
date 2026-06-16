"""Generic direct remote Python selected-profile probe.

The runner copies one local benchmark script to a remote node, launches one
process per requested candidate slot with explicit ``CUDA_VISIBLE_DEVICES``,
parses ``rate=<x> <unit>/s`` progress lines, and emits a service-cache
compatible summary.  It is intentionally outside ``skill/scheduler.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z]+)/s\b")


def build_remote_python_selected_profile_probe(
    *,
    run_id: str,
    node: str,
    gpus: list[int],
    profiles: list[int],
    python_bin: str,
    local_script: str,
    remote_script: str,
    cwd: str,
    benchmark_name: str,
    script_args: str,
    timeout_s: int = 240,
    env: list[str] | None = None,
    launch_delay_s: float = 0.0,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _run(["ssh", node, f"mkdir -p {shlex.quote(cwd)} {shlex.quote(str(Path(remote_script).parent))}"], raw_dir / "remote_mkdir")
    _run(["rsync", "-az", str(Path(local_script)), f"{node}:{remote_script}"], raw_dir / "rsync_script")
    summaries: dict[int, dict[str, Any]] = {}
    for profile in profiles:
        summaries[int(profile)] = _measure_profile(
            run_id=run_id,
            run_dir=run_dir,
            node=node,
            gpus=gpus,
            profile=int(profile),
            python_bin=python_bin,
            remote_script=remote_script,
            cwd=cwd,
            benchmark_name=benchmark_name,
            script_args=script_args,
            timeout_s=timeout_s,
            env=env or [],
            launch_delay_s=launch_delay_s,
        )
    result = {
        "gate": "remote_python_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "gpus": list(gpus),
        "profiles": list(profiles),
        "benchmark_name": benchmark_name,
        "summary_paths": [
            str(report_dir / f"profile_{int(profile)}_per_gpu_summary.json")
            for profile in profiles
        ],
        "rate_count": sum(len(summary.get("rates_unit_s") or []) for summary in summaries.values()),
        "pass": all(bool(summary.get("placement_valid")) for summary in summaries.values()),
    }
    (report_dir / "probe_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _measure_profile(
    *,
    run_id: str,
    run_dir: Path,
    node: str,
    gpus: list[int],
    profile: int,
    python_bin: str,
    remote_script: str,
    cwd: str,
    benchmark_name: str,
    script_args: str,
    timeout_s: int,
    env: list[str],
    launch_delay_s: float,
) -> dict[str, Any]:
    phase = f"profile_{int(profile)}_per_gpu"
    raw_dir = run_dir / "raw" / phase
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_args = shlex.split(script_args or "")
    procs: list[tuple[int, int, subprocess.Popen[str]]] = []
    for gpu in gpus:
        for idx in range(profile):
            label = f"{run_id}-{benchmark_name}-{phase}-gpu{gpu}-{idx}"
            arg_text = " ".join(shlex.quote(x) for x in [*base_args, "--label", label])
            env_text = " ".join(shlex.quote(x) for x in env if x.strip())
            remote = (
                f"cd {shlex.quote(cwd)} && "
                f"CUDA_VISIBLE_DEVICES={int(gpu)} "
                f"PYTHONUNBUFFERED=1 "
                f"{env_text + ' ' if env_text else ''}"
                f"timeout --foreground {int(timeout_s)}s "
                f"{shlex.quote(python_bin)} -u {shlex.quote(remote_script)} {arg_text}"
            )
            proc = subprocess.Popen(
                ["ssh", node, remote],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            procs.append((int(gpu), idx, proc))
            if float(launch_delay_s) > 0.0:
                time.sleep(float(launch_delay_s))

    rows: list[dict[str, Any]] = []
    for gpu, idx, proc in procs:
        try:
            output, _ = proc.communicate(timeout=int(timeout_s) + 45)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        log_path = raw_dir / f"gpu{gpu}_{idx}.log"
        log_path.write_text(output or "", encoding="utf-8")
        rate, unit = _last_rate(output or "")
        rows.append({
            "gpu": gpu,
            "idx": idx,
            "returncode": proc.returncode,
            "rate": rate,
            "unit": unit,
            "log_path": str(log_path),
        })

    rates = [float(row["rate"]) for row in rows if float(row["rate"]) > 0.0]
    expected = {str(int(gpu)): int(profile) for gpu in gpus}
    actual: dict[str, int] = {}
    for row in rows:
        if float(row["rate"]) > 0.0:
            key = str(int(row["gpu"]))
            actual[key] = actual.get(key, 0) + 1
    units = sorted({str(row.get("unit") or "unit") for row in rows if float(row["rate"]) > 0.0})
    summary = {
        "phase": phase,
        "probe": "remote_python_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "benchmark_name": benchmark_name,
        "gpus": list(gpus),
        "profile": int(profile),
        "timeout_s": int(timeout_s),
        "elapsed_wall_s": time.time() - started,
        "running_count": len(rows),
        "running_with_rate_count": len(rates),
        "per_gpu_running": actual,
        "expected_per_gpu_running": expected,
        "rates_unit_s": rates,
        "rate_units": units or ["unit"],
        "aggregate_active_rate_unit_s": sum(rates),
        "mean_active_rate_unit_s": mean(rates) if rates else 0.0,
        "measurement_aggregate_rate_unit_s_median": sum(rates),
        "blocked_count": 0,
        "eviction_count": 0,
        "placement_valid": actual == expected,
        "placement_issues": [] if actual == expected else [f"actual per-GPU running {actual} != expected {expected}"],
        "status_counts": {"sampled": len(rates), "missing_rate": max(0, len(rows) - len(rates))},
        "capacity_boundary": False,
        "rows": rows,
    }
    summary_path = report_dir / f"{phase}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{phase}_summary.md").write_text(_markdown(summary, summary_path), encoding="utf-8")
    return summary


def _run(cmd: list[str], prefix: Path) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_suffix(".stdout")).write_text(proc.stdout or "", encoding="utf-8")
    (prefix.with_suffix(".stderr")).write_text(proc.stderr or "", encoding="utf-8")
    (prefix.with_suffix(".meta.json")).write_text(
        json.dumps({"cmd": cmd, "returncode": proc.returncode}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}")


def _last_rate(output: str) -> tuple[float, str]:
    matches = RATE_RE.findall(output or "")
    if not matches:
        return 0.0, ""
    rate, unit = matches[-1]
    return float(rate), str(unit)


def _markdown(summary: Mapping[str, Any], summary_path: Path) -> str:
    return "\n".join([
        "# Remote Python Selected-Profile Probe",
        "",
        f"Summary: `{summary_path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `benchmark_name` | `{summary.get('benchmark_name')}` |",
        f"| `phase` | `{summary.get('phase')}` |",
        f"| `node` | `{summary.get('node')}` |",
        f"| `running_count` | {summary.get('running_count')} |",
        f"| `running_with_rate_count` | {summary.get('running_with_rate_count')} |",
        f"| `aggregate_active_rate_unit_s` | {float(summary.get('aggregate_active_rate_unit_s') or 0.0):.6g} |",
        f"| `mean_active_rate_unit_s` | {float(summary.get('mean_active_rate_unit_s') or 0.0):.6g} |",
        f"| `placement_valid` | {str(bool(summary.get('placement_valid'))).lower()} |",
        "",
    ])


def _cmd_build(args: argparse.Namespace) -> int:
    result = build_remote_python_selected_profile_probe(
        run_id=args.run_id,
        node=args.node,
        gpus=[int(x) for x in str(args.gpus).split(",") if x.strip()],
        profiles=[int(x) for x in str(args.profiles).split(",") if x.strip()],
        python_bin=args.python_bin,
        local_script=args.local_script,
        remote_script=args.remote_script,
        cwd=args.cwd,
        benchmark_name=args.benchmark_name,
        script_args=args.script_args,
        timeout_s=args.timeout_s,
        env=args.env,
        launch_delay_s=args.launch_delay_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.remote_python_selected_profile_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Run a Python benchmark script at selected per-GPU profiles")
    build.add_argument("--run-id", required=True)
    build.add_argument("--node", required=True)
    build.add_argument("--gpus", default="0,1")
    build.add_argument("--profiles", required=True)
    build.add_argument("--python-bin", required=True)
    build.add_argument("--local-script", required=True)
    build.add_argument("--remote-script", required=True)
    build.add_argument("--cwd", default="~/scheduleurm_bench_cwd")
    build.add_argument("--benchmark-name", required=True)
    build.add_argument("--script-args", default="")
    build.add_argument("--timeout-s", type=int, default=240)
    build.add_argument("--env", action="append", default=[])
    build.add_argument("--launch-delay-s", type=float, default=0.0)
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
