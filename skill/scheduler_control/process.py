"""Identify scheduleurm control-plane processes for auto-adopt filtering."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Iterable


def is_scheduler_control_cmdline(
    cmdline: str,
    *,
    scheduler_file: str | Path,
    control_subcommands: Iterable[str],
    path_markers: Iterable[str],
) -> bool:
    """Return True when cmdline is a scheduleurm CLI/control invocation."""
    if not cmdline:
        return False
    try:
        parts = shlex.split(cmdline)
    except Exception:
        parts = cmdline.split()
    if len(parts) < 2:
        return False
    self_path = os.path.normpath(str(scheduler_file))
    subcommands = {str(x) for x in control_subcommands}
    markers = tuple(str(x) for x in path_markers)
    for i, part in enumerate(parts[:-1]):
        if os.path.basename(part) != "scheduler.py":
            continue
        norm = os.path.normpath(part)
        known_scheduler_path = (
            norm == self_path
            or any(marker in norm for marker in markers)
            or not os.path.isabs(norm)
        )
        if known_scheduler_path and parts[i + 1] in subcommands:
            return True
    return False


def is_repo_path(path: str, *, repo_root: str | Path) -> bool:
    if not path:
        return False
    try:
        norm = os.path.normpath(path)
        root = os.path.normpath(str(repo_root))
    except Exception:
        return False
    return norm == root or norm.startswith(root.rstrip("/") + "/")


def is_scheduler_control_process(
    cwd: str,
    cmdline: str,
    *,
    repo_root: str | Path,
    scheduler_file: str | Path,
    control_subcommands: Iterable[str],
    path_markers: Iterable[str],
) -> bool:
    """Return True for scheduleurm control/test helper processes.

    Auto-adopt is for user workloads. A long-running pytest, one-off
    ``python -`` diagnostic, or scheduler CLI process in the scheduleurm repo
    should not become a fake ``scheduleurm/auto-adopted`` task.
    """
    if is_scheduler_control_cmdline(
        cmdline,
        scheduler_file=scheduler_file,
        control_subcommands=control_subcommands,
        path_markers=path_markers,
    ):
        return True
    if not cmdline or not is_repo_path(cwd, repo_root=repo_root):
        return False
    try:
        parts = shlex.split(cmdline)
    except Exception:
        parts = cmdline.split()
    if not parts:
        return False
    first = os.path.basename(parts[0])
    if first.startswith("pytest") or first in {"py.test"}:
        return True
    if first == "coverage" and any("pytest" in part for part in parts[1:]):
        return True
    if len(parts) >= 3 and parts[1] == "-m" and parts[2].split(".")[0] in {"pytest", "coverage"}:
        return True
    if first.startswith("python"):
        if len(parts) >= 2 and parts[1] in {"-", "-c"}:
            return True
        if len(parts) >= 3 and parts[1] == "-u" and parts[2] in {"-", "-c"}:
            return True
        if any("pytest" in part for part in parts[1:]):
            return True
        if any(os.path.basename(part) == "test_regression.py" for part in parts[1:]):
            return True
    return False
