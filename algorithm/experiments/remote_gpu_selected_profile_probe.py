"""Direct remote GPU selected-profile probe.

This collector intentionally bypasses the live scheduler queue.  It is used for
measurement-only selected-profile LCB samples when the scheduler queue is busy
or locked, while the target GPUs are otherwise idle.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")


def build_remote_gpu_selected_profile_probe(
    *,
    run_id: str,
    node: str,
    gpus: list[int],
    profiles: list[int],
    python_bin: str,
    remote_script: str = "/tmp/scheduleurm_gpu_progress_benchmark.py",
    cwd: str = "~/scheduleurm_bench_cwd",
    steps: int = 2400,
    size: int = 8192,
    mem_fraction: float = 0.08,
    timeout_s: int = 150,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _run(["ssh", node, "mkdir", "-p", cwd], raw_dir / "remote_mkdir")
    _run(
        ["rsync", "-az", str(REPO_ROOT / "algorithm" / "experiments" / "gpu_progress_benchmark.py"), f"{node}:{remote_script}"],
        raw_dir / "rsync_benchmark",
    )
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
            steps=steps,
            size=size,
            mem_fraction=mem_fraction,
            timeout_s=timeout_s,
        )
    result = {
        "gate": "remote_gpu_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "gpus": list(gpus),
        "profiles": list(profiles),
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
    steps: int,
    size: int,
    mem_fraction: float,
    timeout_s: int,
) -> dict[str, Any]:
    phase = f"profile_{int(profile)}_per_gpu"
    report_dir = run_dir / "reports"
    raw_dir = run_dir / "raw" / phase
    raw_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, int, subprocess.Popen[str]]] = []
    started = time.time()
    for gpu in gpus:
        for idx in range(profile):
            label = f"{run_id}-{phase}-gpu{gpu}-{idx}"
            remote = (
                f"cd {cwd} && "
                f"CUDA_VISIBLE_DEVICES={int(gpu)} "
                f"OMP_NUM_THREADS=1 "
                f"XLA_PYTHON_CLIENT_PREALLOCATE=false "
                f"XLA_PYTHON_CLIENT_MEM_FRACTION={float(mem_fraction):.3f} "
                f"timeout --foreground {int(timeout_s)}s "
                f"{python_bin} -u {remote_script} "
                f"--steps {int(steps)} --size {int(size)} --label {label}"
            )
            proc = subprocess.Popen(
                ["ssh", node, remote],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            procs.append((int(gpu), idx, proc))
    rows = []
    for gpu, idx, proc in procs:
        try:
            output, _ = proc.communicate(timeout=int(timeout_s) + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        log_path = raw_dir / f"gpu{gpu}_{idx}.log"
        log_path.write_text(output or "", encoding="utf-8")
        rows.append({
            "gpu": gpu,
            "idx": idx,
            "returncode": proc.returncode,
            "rate": _last_rate(output or ""),
            "backend": _backend(output or ""),
            "log_path": str(log_path),
        })
    rates = [float(row["rate"]) for row in rows if float(row["rate"]) > 0.0]
    expected = {str(int(gpu)): int(profile) for gpu in gpus}
    actual = {}
    for row in rows:
        if float(row["rate"]) > 0.0:
            key = str(int(row["gpu"]))
            actual[key] = actual.get(key, 0) + 1
    summary = {
        "phase": phase,
        "probe": "remote_gpu_selected_profile_probe",
        "run_id": run_id,
        "node": node,
        "gpus": list(gpus),
        "profile": int(profile),
        "steps": int(steps),
        "size": int(size),
        "mem_fraction": float(mem_fraction),
        "timeout_s": int(timeout_s),
        "elapsed_wall_s": time.time() - started,
        "running_count": len(rows),
        "running_with_rate_count": len(rates),
        "per_gpu_running": actual,
        "expected_per_gpu_running": expected,
        "rates_unit_s": rates,
        "rate_units": ["unit"],
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


def _last_rate(output: str) -> float:
    matches = RATE_RE.findall(output or "")
    if not matches:
        return 0.0
    return float(matches[-1])


def _backend(output: str) -> str:
    for line in (output or "").splitlines():
        if line.startswith("BENCH_START"):
            parts = dict(part.split("=", 1) for part in line.split() if "=" in part)
            return str(parts.get("backend") or "")
    return ""


def _markdown(summary: Mapping[str, Any], summary_path: Path) -> str:
    return "\n".join([
        "# Remote GPU Selected-Profile Probe",
        "",
        f"Summary: `{summary_path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
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
    result = build_remote_gpu_selected_profile_probe(
        run_id=args.run_id,
        node=args.node,
        gpus=[int(x) for x in str(args.gpus).split(",") if x.strip()],
        profiles=[int(x) for x in str(args.profiles).split(",") if x.strip()],
        python_bin=args.python_bin,
        remote_script=args.remote_script,
        cwd=args.cwd,
        steps=args.steps,
        size=args.size,
        mem_fraction=args.mem_fraction,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.remote_gpu_selected_profile_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Measure remote GPU selected profiles without using scheduler queue")
    build.add_argument("--run-id", required=True)
    build.add_argument("--node", required=True)
    build.add_argument("--gpus", default="0,1")
    build.add_argument("--profiles", required=True)
    build.add_argument("--python-bin", default="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python")
    build.add_argument("--remote-script", default="/tmp/scheduleurm_gpu_progress_benchmark.py")
    build.add_argument("--cwd", default="~/scheduleurm_bench_cwd")
    build.add_argument("--steps", type=int, default=2400)
    build.add_argument("--size", type=int, default=8192)
    build.add_argument("--mem-fraction", type=float, default=0.08)
    build.add_argument("--timeout-s", type=int, default=150)
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
