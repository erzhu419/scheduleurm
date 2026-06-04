"""Line-oriented progress wrapper for real workload service-curve runs.

The scheduler watches task logs line by line.  A terminal tqdm bar is useful for
humans but often collapses into carriage-return fragments in remote logs.  This
wrapper preserves the child process output and adds one normalized progress line
whenever a task-native counter/rate can be parsed.
"""
from __future__ import annotations

import argparse
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
) -> str | None:
    obs = parse_progress_line(line, cmd=_command_for_parser(command))
    if obs is None:
        return None
    return build_progress_line(obs, total_override=total_override, unit_override=unit_override)


def _child_return_code_for_shell(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return int(returncode)


def run_wrapped_command(command: Sequence[str], *, total: int | None, unit: str) -> int:
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
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(line, flush=True)
            progress_line = _normalize_child_line(
                line,
                command=command,
                total_override=total,
                unit_override=unit,
            )
            if progress_line:
                print(progress_line, flush=True)
    finally:
        return _child_return_code_for_shell(proc.wait())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="iter")
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    total = int(args.total) if int(args.total or 0) > 0 else None
    return run_wrapped_command(command, total=total, unit=args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
