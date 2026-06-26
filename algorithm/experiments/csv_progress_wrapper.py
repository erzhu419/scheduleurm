"""Run a command while reporting progress from diagnostics CSV files."""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Sequence


def _format_hms(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _tqdm_progress_line(
    *,
    unit: str,
    current: int,
    total: int,
    elapsed_s: float,
    rate: float,
) -> str | None:
    if current < 0 or total <= 0 or current > total or rate <= 0:
        return None
    remaining_s = max(0.0, float(total - current) / max(float(rate), 1e-12))
    pct = 100.0 * float(current) / float(total)
    filled = int(round(10.0 * float(current) / float(total)))
    bar = "#" * max(0, min(10, filled)) + "-" * max(0, 10 - min(10, filled))
    return (
        f"ScheduleurmTqdm {unit}: {pct:5.1f}%|{bar}| "
        f"{current}/{total} [{_format_hms(elapsed_s)}<{_format_hms(remaining_s)}, "
        f"{float(rate):.3g}{unit}/s]"
    )


def _count_csv_units(patterns: Sequence[str]) -> int:
    total = 0
    seen: set[str] = set()
    for pattern in patterns:
        for raw in glob.glob(pattern, recursive=True):
            path = str(Path(raw))
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for line in f if line.strip())
            except OSError:
                continue
            if lines > 1:
                total += lines - 1
    return total


def _progress_monitor(
    *,
    patterns: Sequence[str],
    unit: str,
    total: int,
    poll_s: float,
    start_time: float,
    stop: threading.Event,
) -> None:
    last_current = -1
    while not stop.wait(max(0.2, float(poll_s))):
        current = _count_csv_units(patterns)
        if current <= last_current:
            continue
        last_current = current
        elapsed = max(1e-9, time.monotonic() - float(start_time))
        rate = current / elapsed
        parts = [
            "ScheduleurmProgress",
            unit.capitalize(),
            f"{current}/{int(total)}" if total > 0 else str(current),
        ]
        if rate > 0:
            parts.append(f"rate={rate:.9g} {unit}/s")
            if total > 0:
                eta_s = max(0, int(total) - int(current)) / rate
                parts.append(f"ETA {eta_s:.1f}s")
            parts.append(f"seconds_per_{unit}={1.0 / rate:.6g}")
        parts.append("source=csv_poll")
        print(" ".join(parts), flush=True)
        tqdm_line = _tqdm_progress_line(
            unit=unit,
            current=int(current),
            total=int(total),
            elapsed_s=elapsed,
            rate=rate,
        )
        if tqdm_line:
            print(tqdm_line, flush=True)


def _child_return_code_for_shell(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return int(returncode)


def run_wrapped_command(
    command: Sequence[str],
    *,
    patterns: Sequence[str],
    total: int,
    unit: str,
    poll_s: float,
) -> int:
    if not command:
        raise ValueError("wrapped command is empty")
    if not patterns:
        raise ValueError("at least one --csv-glob is required")

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    stop = threading.Event()
    start_time = time.monotonic()
    monitor = threading.Thread(
        target=_progress_monitor,
        kwargs={
            "patterns": list(patterns),
            "unit": unit,
            "total": int(total),
            "poll_s": float(poll_s),
            "start_time": start_time,
            "stop": stop,
        },
        daemon=True,
    )
    monitor.start()
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
            print(raw.rstrip("\n"), flush=True)
    finally:
        rc = _child_return_code_for_shell(proc.wait())
        stop.set()
        current = _count_csv_units(patterns)
        if current > 0:
            print(
                "ScheduleurmProgress "
                f"{unit.capitalize()} {current}/{int(total)} "
                "source=csv_poll_final",
                flush=True,
            )
            elapsed = max(1e-9, time.monotonic() - start_time)
            rate = float(current) / elapsed
            tqdm_line = _tqdm_progress_line(
                unit=unit,
                current=int(current),
                total=int(total),
                elapsed_s=elapsed,
                rate=rate,
            )
            if tqdm_line:
                print(tqdm_line, flush=True)
        return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="episode")
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--csv-glob", action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    return run_wrapped_command(
        command,
        patterns=args.csv_glob,
        total=int(args.total or 0),
        unit=str(args.unit or "episode").lower(),
        poll_s=float(args.poll_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
