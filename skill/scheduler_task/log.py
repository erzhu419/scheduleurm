"""Task log tailing helpers with scheduler-specific I/O injected by the caller."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Callable, Optional


TailResult = tuple[bool, str, str]


def missing_log_path_message(task: dict) -> str:
    task_id = task.get("id")
    if task.get("auto_adopted") or task.get("adopted") or task.get("origin") == "external":
        return (
            f"task {task_id} has no scheduler-captured log_path; "
            "it was adopted from an already-running external process, so stdout/stderr "
            "were not redirected through scheduleurm"
        )
    return f"task {task_id} has no log_path (not launched yet or launch metadata was not committed)"


def normalize_tail_lines(lines: int, default: int = 80, max_lines: int = 10000) -> int:
    try:
        value = int(lines or default)
    except Exception:
        value = default
    return max(1, min(max_lines, value))


def tail_local_log(log_path: str, lines: int,
                   runner: Optional[Callable[..., subprocess.CompletedProcess]] = None) -> TailResult:
    lp = Path(log_path)
    if not lp.exists():
        return False, log_path, f"log file not found: {log_path}"
    run = runner or subprocess.run
    result = run(
        ["tail", "-n", str(lines), str(lp)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return False, log_path, (result.stderr or result.stdout or "tail failed").strip()
    return True, log_path, result.stdout


def build_windows_tail_script(win_path: str, lines: int, ps_quote: Callable[[str], str]) -> str:
    return rf'''
$path = {ps_quote(win_path)}
if (-not (Test-Path -LiteralPath $path)) {{ Write-Error "log file not found: $path"; exit 2 }}
Get-Content -LiteralPath $path -Tail {lines} -ErrorAction Stop
'''


def tail_task_log(
    task: dict,
    lines: int = 80,
    *,
    nodes: dict,
    node_is_windows: Callable[[str], bool],
    windows_path_for_task: Callable[[dict, str], str],
    ps_quote: Callable[[str], str],
    run_windows_ps: Callable[..., tuple[int, str, str]],
    run_on: Callable[..., tuple[int, str, str]],
    local_runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> TailResult:
    log_path = task.get("log_path")
    node = task.get("node")
    if not log_path:
        return False, "", missing_log_path_message(task)
    lines = normalize_tail_lines(lines)
    try:
        if not node or nodes.get(node, {}).get("host") is None:
            return tail_local_log(log_path, lines, runner=local_runner)
        if node_is_windows(node):
            win_path = windows_path_for_task(task, log_path)
            script = build_windows_tail_script(win_path, lines, ps_quote)
            rc, out, err = run_windows_ps(node, script, timeout=30, check=False)
        else:
            rc, out, err = run_on(
                node,
                f"tail -n {lines} {shlex.quote(log_path)}",
                timeout=30,
                check=False,
            )
        if rc != 0:
            return False, log_path, (err or out or f"tail failed rc={rc}").strip()
        return True, log_path, out
    except Exception as e:
        return False, log_path, str(e)
