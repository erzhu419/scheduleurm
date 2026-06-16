"""Line-oriented progress wrapper for real workload service-curve runs.

The scheduler watches task logs line by line.  A terminal tqdm bar is useful for
humans but often collapses into carriage-return fragments in remote logs.  This
wrapper preserves the child process output and adds one normalized progress line
whenever a task-native counter/rate can be parsed.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.experiments.progress_units import ProgressObservation, canonical_unit, parse_progress_line


def _command_for_parser(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def build_progress_line(
    obs: ProgressObservation,
    *,
    total_override: int | None = None,
    unit_override: str | None = None,
) -> str | None:
    """Render a scheduler-stable progress line from a parsed child log line."""

    current = obs.current
    total = int(total_override) if total_override and int(total_override) > 0 else obs.total
    rate = obs.rate_per_s
    unit = canonical_unit(unit_override or obs.unit)
    if current is None and rate is None:
        return None
    if current is not None and total is not None and int(current) > int(total):
        return None

    parts = ["ScheduleurmProgress"]
    label = unit.capitalize() if unit else "Unit"
    if current is not None and total is not None:
        parts += [label, f"{int(current)}/{int(total)}"]
    elif current is not None:
        parts += [label, str(int(current))]
    else:
        parts += [label, "unknown"]

    if rate is not None and float(rate) > 0:
        rate_f = float(rate)
        parts.append(f"rate={rate_f:.9g} {unit}/s")
        if current is not None and total is not None:
            remaining = max(0, int(total) - int(current))
            eta_s = remaining / rate_f
            parts.append(f"ETA {eta_s:.1f}s")
        if obs.seconds_per_unit is not None:
            parts.append(f"seconds_per_{unit}={float(obs.seconds_per_unit):.6g}")
    parts.append(f"source={obs.source}")
    return " ".join(parts)


def _normalize_child_line(
    line: str,
    *,
    command: Sequence[str],
    total_override: int | None,
    unit_override: str | None,
) -> tuple[str | None, ProgressObservation | None]:
    obs = parse_progress_line(line, cmd=_command_for_parser(command))
    if obs is None:
        return None, None
    return build_progress_line(obs, total_override=total_override, unit_override=unit_override), obs


def _stable_rate_decision(
    rates: Sequence[float],
    *,
    windows: int,
    min_samples: int,
    max_cv: float,
    max_last_two_rel_delta: float,
    skip_samples: int,
) -> dict[str, object]:
    usable = [float(x) for x in rates[int(max(0, skip_samples)):] if float(x) > 0.0]
    needed = max(int(min_samples), int(windows), 1)
    if len(usable) < needed:
        return {
            "ready": False,
            "reason": f"need_at_least_{needed}_positive_rate_samples",
            "usable_samples": len(usable),
            "tail": usable[-int(max(1, windows)):],
        }
    tail = usable[-int(max(1, windows)):]
    avg = sum(tail) / len(tail)
    var = sum((x - avg) ** 2 for x in tail) / len(tail)
    cv = math.sqrt(var) / max(abs(avg), 1e-12)
    rel = (
        abs(tail[-1] - tail[-2]) / max(abs(tail[-2]), 1e-12)
        if len(tail) >= 2
        else 0.0
    )
    return {
        "ready": cv <= float(max_cv) and rel <= float(max_last_two_rel_delta),
        "tail": tail,
        "mean_rate": avg,
        "last_rate": tail[-1],
        "cv": cv,
        "last_two_relative_delta": rel,
        "usable_samples": len(usable),
        "rule": (
            f"{len(tail)} windows after {int(max(0, skip_samples))} skipped samples, "
            f"cv<={float(max_cv):.6g}, last-two relative delta<={float(max_last_two_rel_delta):.6g}"
        ),
    }


def _stable_rate_line(decision: dict[str, object], *, unit: str) -> str:
    tail = ",".join(f"{float(x):.9g}" for x in decision.get("tail", []) or [])
    return (
        "ScheduleurmStableRate "
        f"unit={canonical_unit(unit)} "
        f"mean_rate={float(decision.get('mean_rate') or 0.0):.9g} "
        f"last_rate={float(decision.get('last_rate') or 0.0):.9g} "
        f"cv={float(decision.get('cv') or 0.0):.9g} "
        f"last_two_relative_delta={float(decision.get('last_two_relative_delta') or 0.0):.9g} "
        f"samples={int(decision.get('usable_samples') or 0)} "
        f"tail={tail} "
        f"rule={str(decision.get('rule') or '').replace(' ', '_')}"
    )


def _child_return_code_for_shell(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return int(returncode)


def _terminate_child(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_wrapped_command(
    command: Sequence[str],
    *,
    total: int | None,
    unit: str,
    terminate_on_stable: bool = False,
    stable_windows: int = 3,
    min_rate_samples: int = 3,
    stable_cv: float = 0.08,
    stable_rel_delta: float = 0.05,
    stable_skip_samples: int = 0,
) -> int:
    if not command:
        raise ValueError("wrapped command is empty")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    assert proc.stdout is not None
    rates: list[float] = []
    stopped_on_stable = False
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(line, flush=True)
        progress_line, obs = _normalize_child_line(
            line,
            command=command,
            total_override=total,
            unit_override=unit,
        )
        if progress_line:
            print(progress_line, flush=True)
        if obs is None or obs.rate_per_s is None or float(obs.rate_per_s) <= 0.0:
            continue
        rates.append(float(obs.rate_per_s))
        if not terminate_on_stable:
            continue
        decision = _stable_rate_decision(
            rates,
            windows=stable_windows,
            min_samples=min_rate_samples,
            max_cv=stable_cv,
            max_last_two_rel_delta=stable_rel_delta,
            skip_samples=stable_skip_samples,
        )
        if bool(decision.get("ready")):
            print(_stable_rate_line(decision, unit=unit), flush=True)
            _terminate_child(proc)
            stopped_on_stable = True
            break
    rc = _child_return_code_for_shell(proc.wait())
    return 0 if stopped_on_stable else rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="iter")
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--terminate-on-stable", action="store_true")
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--min-rate-samples", type=int, default=3)
    parser.add_argument("--stable-cv", type=float, default=0.08)
    parser.add_argument("--stable-rel-delta", type=float, default=0.05)
    parser.add_argument("--stable-skip-samples", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    total = int(args.total) if int(args.total or 0) > 0 else None
    return run_wrapped_command(
        command,
        total=total,
        unit=args.unit,
        terminate_on_stable=bool(args.terminate_on_stable),
        stable_windows=args.stable_windows,
        min_rate_samples=args.min_rate_samples,
        stable_cv=args.stable_cv,
        stable_rel_delta=args.stable_rel_delta,
        stable_skip_samples=args.stable_skip_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
