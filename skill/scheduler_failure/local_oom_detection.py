"""Local syslog OOM-kill post-processing for terminal task diagnosis."""

from __future__ import annotations

import datetime
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class LocalOomKillDetectionDeps:
    syslog_path: str = "/var/log/syslog"
    path_exists: Callable[[str], bool] = lambda path: Path(path).exists()
    check_output: Callable[..., str] = subprocess.check_output
    now: Callable[[], float] = __import__("time").time
    current_year: Callable[[], int] = lambda: datetime.datetime.now().year


def detect_oom_kills_local(state: dict, *, deps: LocalOomKillDetectionDeps) -> list:
    """Flip local done tasks to failed when they overlap a recent syslog OOM kill."""
    if not deps.path_exists(deps.syslog_path):
        return []
    try:
        out = deps.check_output(
            ["tail", "-5000", deps.syslog_path],
            text=True,
            errors="replace",
            timeout=5,
        )
    except Exception:
        return []
    cutoff = deps.now() - 1800
    pattern = re.compile(r"^(\w+\s+\d+\s+\d+:\d+:\d+).*Out of memory: Killed process")
    oom_times = []
    year = deps.current_year()
    for line in out.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            ts = datetime.datetime.strptime(f"{year} {match.group(1)}", "%Y %b %d %H:%M:%S").timestamp()
        except Exception:
            continue
        if ts >= cutoff:
            oom_times.append(ts)
    if not oom_times:
        return []

    flipped = []
    for task in state.get("tasks", []):
        if task.get("node") != "local":
            continue
        if task.get("status") != "done":
            continue
        if task.get("auto_adopted"):
            continue
        started = task.get("started_at") or 0
        finished = task.get("finished_at") or 0
        if finished < cutoff:
            continue
        if not any(started - 30 <= oom_t <= finished + 60 for oom_t in oom_times):
            continue
        diag = task.get("_diagnosis") or {}
        if diag.get("success_marker"):
            continue
        if int(task.get("peak_vram_mb") or 0) > 100:
            continue
        if task.get("_oom_flipped"):
            continue
        task["status"] = "failed"
        task["_oom_flipped"] = True
        task["last_block_reason"] = (
            "ex-post OOM: terminated within local OOM-kill window without "
            f"entering training (peak_vram={task.get('peak_vram_mb')}MB)"
        )
        diag = dict(diag)
        diag["is_crash"] = True
        diag["reason"] = (diag.get("reason") or "") + " | ex-post: syslog OOM in window"
        task["_diagnosis"] = diag
        flipped.append(task)
    return flipped
