#!/usr/bin/env python3
"""Start a dead watcher and restart a live-but-stale watcher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler_watch.health import decide_watcher_health  # noqa: E402


def _systemctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _main_pid(service: str) -> int:
    result = _systemctl("show", service, "--property=MainPID", "--value")
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


def _process_age_s(pid: int) -> float:
    if pid <= 0:
        return 0.0
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
        start_ticks = int(stat_fields[21])
        uptime_s = float(Path("/proc/uptime").read_text().split()[0])
        ticks_per_s = int(os.sysconf("SC_CLK_TCK"))
        return max(0.0, uptime_s - start_ticks / ticks_per_s)
    except (OSError, ValueError, IndexError):
        return 0.0


def _load_state(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="scheduler.service")
    parser.add_argument(
        "--state",
        type=Path,
        default=Path.home() / ".claude/scheduler/.watcher_state.json",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=float(os.environ.get("SCHEDULEURM_WATCHDOG_STALE_S", "600")),
    )
    args = parser.parse_args(argv)

    active = _systemctl("is-active", "--quiet", args.service).returncode == 0
    pid = _main_pid(args.service) if active else 0
    process_alive = pid > 0 and Path(f"/proc/{pid}").exists()
    decision = decide_watcher_health(
        service_active=active,
        main_pid=pid,
        process_alive=process_alive,
        process_age_s=_process_age_s(pid),
        state=_load_state(args.state),
        now=time.time(),
        stale_after_s=args.stale_after,
    )
    if decision.action == "none":
        return 0
    print(
        f"scheduler keepalive: {decision.action} {args.service}: "
        f"{decision.reason}; phase={decision.phase} age={decision.age_s:.0f}s",
        flush=True,
    )
    if decision.action == "start":
        _systemctl("reset-failed", args.service)
        return _systemctl("start", args.service).returncode
    return _systemctl("restart", args.service).returncode


if __name__ == "__main__":
    raise SystemExit(main())
