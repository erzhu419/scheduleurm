"""Remote resident-plus-marginal probe runner for corner-case validation."""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import re
import shlex
import time
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
REMOTE_DIR = "/tmp/scheduleurm_corner_case_probe"
BENCHMARK_FILES = [
    "algorithm/experiments/gpu_progress_benchmark.py",
    "algorithm/experiments/gpu_multigpu_memory_progress_benchmark.py",
    "algorithm/experiments/torch_cnn_progress_benchmark.py",
    "algorithm/experiments/torch_llm_progress_benchmark.py",
    "algorithm/experiments/cpu_progress_benchmark.py",
    "algorithm/experiments/cpu_parallel_progress_benchmark.py",
]
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")
ETA_RE = re.compile(r"\bETA\s+([0-9]+(?:\.[0-9]+)?)s\b")


def run_remote_marginal_probe(
    *,
    run_id: str,
    scenario_id: str,
    node: str,
    probe_cmd: str,
    background_cmd: str = "",
    warmup_s: float = 8.0,
    probe_timeout_s: int = 120,
    background_timeout_s: int = 180,
    remote_dir: str = REMOTE_DIR,
    algorithm_mode: str = "theorem_maxweight_v1",
    hard_rule_mode: str = "clean_bench",
) -> dict[str, Any]:
    scheduler = _load_scheduler_module()
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw" / scenario_id
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _deploy_benchmarks(scheduler, node=node, remote_dir=remote_dir, raw_dir=raw_dir)
    remote_shell = _remote_shell(
        remote_dir=remote_dir,
        probe_cmd=probe_cmd,
        background_cmd=background_cmd,
        warmup_s=warmup_s,
        probe_timeout_s=probe_timeout_s,
        background_timeout_s=background_timeout_s,
    )
    rc, out, err = scheduler.run_on(node, remote_shell, timeout=max(background_timeout_s, probe_timeout_s) + 45, check=False)
    combined = (out or "") + ("\n__LOCAL_STDERR__\n" + err if err else "")
    (raw_dir / "remote_combined.log").write_text(combined, encoding="utf-8")
    bg_log = _between(out or "", "__SCHEDULEURM_BG_BEGIN__", "__SCHEDULEURM_BG_END__")
    probe_log = _between(out or "", "__SCHEDULEURM_PROBE_BEGIN__", "__SCHEDULEURM_PROBE_END__")
    (raw_dir / "background.log").write_text(bg_log, encoding="utf-8")
    (raw_dir / "probe.log").write_text(probe_log, encoding="utf-8")
    bg_rates = _rates(bg_log)
    probe_rates = _rates(probe_log)
    report = {
        "gate": "remote_corner_case_probe",
        "run_id": run_id,
        "scenario_id": scenario_id,
        "node": node,
        "algorithm_mode": algorithm_mode,
        "hard_rule_mode": hard_rule_mode,
        "remote_dir": remote_dir,
        "background_cmd": background_cmd,
        "probe_cmd": probe_cmd,
        "remote_wrapper_returncode": rc,
        "background_rate_count": len(bg_rates),
        "probe_rate_count": len(probe_rates),
        "background_last_rate_step_s": bg_rates[-1] if bg_rates else 0.0,
        "probe_last_rate_step_s": probe_rates[-1] if probe_rates else 0.0,
        "probe_mean_rate_step_s": mean(probe_rates) if probe_rates else 0.0,
        "probe_eta_count": len(_etas(probe_log)),
        "stable_eta": _stable(probe_rates),
        "pass": rc == 0 and bool(probe_rates) and "ETA " in probe_log,
        "raw_paths": {
            "combined": str(raw_dir / "remote_combined.log"),
            "background": str(raw_dir / "background.log"),
            "probe": str(raw_dir / "probe.log"),
        },
    }
    report_path = report_dir / f"{scenario_id}_remote_corner_case_probe.json"
    md_path = report_dir / f"{scenario_id}_remote_corner_case_probe.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report, report_path), encoding="utf-8")
    return report


def _deploy_benchmarks(scheduler: Any, *, node: str, remote_dir: str, raw_dir: Path) -> None:
    scheduler.run_on(node, f"mkdir -p {shlex.quote(remote_dir)}", timeout=20, check=True)
    for rel in BENCHMARK_FILES:
        local = REPO_ROOT / rel
        data = base64.b64encode(local.read_bytes()).decode("ascii")
        remote = f"{remote_dir.rstrip('/')}/{Path(rel).name}"
        cmd = (
            f"base64 -d > {shlex.quote(remote)} <<'__SCHEDULEURM_B64__'\n"
            f"{data}\n"
            "__SCHEDULEURM_B64__\n"
            f"chmod 755 {shlex.quote(remote)}"
        )
        rc, out, err = scheduler.run_on(node, cmd, timeout=30, check=False)
        (raw_dir / f"deploy_{Path(rel).name}.log").write_text((out or "") + (err or ""), encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"deploy failed for {rel} on {node}: {err[:300]}")


def _remote_shell(
    *,
    remote_dir: str,
    probe_cmd: str,
    background_cmd: str,
    warmup_s: float,
    probe_timeout_s: int,
    background_timeout_s: int,
) -> str:
    lines = [
        "set +e",
        f"cd {shlex.quote(remote_dir)}",
        "rm -f background.log probe.log bg.pid",
    ]
    if background_cmd.strip():
        bg = f"timeout --foreground {int(background_timeout_s)}s bash -lc {shlex.quote(background_cmd)}"
        lines.append(f"({bg}) > background.log 2>&1 & echo $! > bg.pid")
        lines.append(f"sleep {float(warmup_s):.3f}")
    probe = f"timeout --foreground {int(probe_timeout_s)}s bash -lc {shlex.quote(probe_cmd)}"
    lines.extend([
        f"({probe}) > probe.log 2>&1",
        "PROBE_RC=$?",
        "if [ -f bg.pid ]; then kill $(cat bg.pid) >/dev/null 2>&1 || true; wait $(cat bg.pid) >/dev/null 2>&1 || true; fi",
        "echo __SCHEDULEURM_BG_BEGIN__",
        "cat background.log 2>/dev/null || true",
        "echo __SCHEDULEURM_BG_END__",
        "echo __SCHEDULEURM_PROBE_BEGIN__",
        "cat probe.log 2>/dev/null || true",
        "echo __SCHEDULEURM_PROBE_END__",
        "exit $PROBE_RC",
    ])
    return "\n".join(lines)


def _rates(text: str) -> list[float]:
    return [float(x) for x in RATE_RE.findall(text or "")]


def _etas(text: str) -> list[float]:
    return [float(x) for x in ETA_RE.findall(text or "")]


def _stable(rates: list[float]) -> dict[str, Any]:
    tail = [float(x) for x in rates[-3:] if float(x) > 0.0]
    if len(tail) < 3:
        return {"ready": False, "reason": "need_at_least_3_rate_windows", "tail": tail}
    avg = mean(tail)
    var = sum((x - avg) ** 2 for x in tail) / len(tail)
    cv = math.sqrt(var) / max(abs(avg), 1e-12)
    rel = abs(tail[-1] - tail[-2]) / max(abs(tail[-2]), 1e-12)
    return {
        "ready": cv <= 0.08 and rel <= 0.05,
        "tail": tail,
        "cv": cv,
        "last_two_relative_delta": rel,
        "rule": "3 windows, cv<=0.08, last-two relative delta<=0.05",
    }


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _load_scheduler_module():
    path = REPO_ROOT / "skill" / "scheduler.py"
    spec = importlib.util.spec_from_file_location("scheduleurm_scheduler_for_remote_corner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown(report: Mapping[str, Any], path: Path) -> str:
    return "\n".join([
        "# Remote Corner-Case Probe",
        "",
        f"JSON artifact: `{path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| scenario | `{report.get('scenario_id')}` |",
        f"| node | `{report.get('node')}` |",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| background_rate_count | {report.get('background_rate_count')} |",
        f"| probe_rate_count | {report.get('probe_rate_count')} |",
        f"| probe_last_rate_step_s | {float(report.get('probe_last_rate_step_s') or 0.0):.6g} |",
        f"| stable_eta_ready | {str(bool((report.get('stable_eta') or {}).get('ready'))).lower()} |",
        "",
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.remote_corner_case_probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-marginal")
    run.add_argument("--run-id", required=True)
    run.add_argument("--scenario-id", required=True)
    run.add_argument("--node", required=True)
    run.add_argument("--probe-cmd", required=True)
    run.add_argument("--background-cmd", default="")
    run.add_argument("--warmup-s", type=float, default=8.0)
    run.add_argument("--probe-timeout-s", type=int, default=120)
    run.add_argument("--background-timeout-s", type=int, default=180)
    run.add_argument("--remote-dir", default=REMOTE_DIR)
    run.add_argument("--algorithm-mode", default="theorem_maxweight_v1")
    run.add_argument("--hard-rule-mode", default="clean_bench")
    run.set_defaults(func=_cmd_run_marginal)
    return parser


def _cmd_run_marginal(args: argparse.Namespace) -> int:
    report = run_remote_marginal_probe(
        run_id=args.run_id,
        scenario_id=args.scenario_id,
        node=args.node,
        probe_cmd=args.probe_cmd,
        background_cmd=args.background_cmd,
        warmup_s=args.warmup_s,
        probe_timeout_s=args.probe_timeout_s,
        background_timeout_s=args.background_timeout_s,
        remote_dir=args.remote_dir,
        algorithm_mode=args.algorithm_mode,
        hard_rule_mode=args.hard_rule_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
